"""Batched generation engine.

Features: thinking-budget forcing (truncate <think> at B tokens, force
"</think>" + the answer prefix), optional J-space ablation with clean-pass
exemptions, per-sequence reproducible sampling (own RNG per sequence, so
results don't depend on batch composition), batch compaction as sequences
finish, and KV-budget eviction: when the caches outgrow `kv_budget_gb`, the
longest-running sequences are evicted with their full state (token ids, RNG
state, counters) and can be resumed exactly in a later, smaller-batch wave.
"""
from __future__ import annotations

import random
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from .model import ANSWER_PREFIX, END_THINK_ID, IM_END_ID, PAD_ID, THINK_SAMPLING

FORCE_END_THINK = (198, 151668)   # "\n</think>"
THINK, ANSWER, DONE = 0, 1, 2
KV_BYTES_PER_TOKEN = 36 * 2 * 8 * 128 * 2   # layers * (K,V) * kv_heads * head_dim * bf16


@dataclass
class GenRequest:
    prompt: str                     # fully templated; ends inside <think> if thinking
    budget: int                     # max thinking tokens B
    seed: int
    thinking: bool = True
    max_answer_tokens: int = 64
    meta: dict = field(default_factory=dict)
    # --- continuation state (set on eviction, consumed on resume) ---
    prior_ids: list = field(default_factory=list)
    state: dict | None = None       # phase/depth/counters/forced/rng state


@dataclass
class GenResult:
    completion: str
    answer_text: str
    think_tokens: int
    answer_tokens: int
    think_truncated: bool
    finished: bool
    meta: dict
    ids: list = field(default_factory=list)


def _sample_rows(logits: torch.Tensor, params: dict, rngs: list) -> list[int]:
    t, k, p = params["temperature"], params["top_k"], params["top_p"]
    scaled = logits.float() / t
    topv, topi = scaled.topk(k, dim=-1)
    probs = torch.softmax(topv, dim=-1)
    sortp, sorti = probs.sort(dim=-1, descending=True)
    cum = sortp.cumsum(-1)
    keep = (cum - sortp) < p          # first token always kept
    sortp = sortp * keep
    sortp = sortp / sortp.sum(-1, keepdim=True)
    sortp = sortp.cpu()
    src = topi.gather(-1, sorti).cpu()
    out = []
    for row, rng in enumerate(rngs):
        r = rng.random()
        c = 0.0
        j = 0
        pr = sortp[row]
        for j in range(pr.numel()):
            c += float(pr[j])
            if r < c:
                break
        out.append(int(src[row, j]))
    return out


def _clean_topk(model, hidden: torch.Tensor, k: int) -> torch.Tensor:
    outs = []
    for s in range(0, hidden.shape[1], 64):
        h = hidden[:, s:s + 64]
        logits = model.lm_head(model.model.norm(h))
        outs.append(logits.topk(k, dim=-1).indices)
    return torch.cat(outs, dim=1)


def _forward(model, ids, attn, pos, cache, hidden=False):
    kw = dict(input_ids=ids, attention_mask=attn, position_ids=pos,
              past_key_values=cache, use_cache=True)
    if hidden:
        kw["output_hidden_states"] = True
    try:
        return model(**kw, logits_to_keep=1)
    except TypeError:
        return model(**kw)


class _Seq:
    """Per-sequence generation state."""

    def __init__(self, req: GenRequest, idx: int):
        self.req = req
        self.idx = idx                       # index into the requests list
        self.gen_ids: list[int] = list(req.prior_ids)
        st = req.state
        if st is not None:
            self.phase = st["phase"]
            self.depth = st["depth"]
            self.think_ct = st["think_ct"]
            self.ans_ct = st["ans_ct"]
            self.truncated = st["truncated"]
            self.forced = deque(st["forced"])
            self.rng = random.Random()
            self.rng.setstate(st["rng"])
        else:
            self.phase = THINK if req.thinking else ANSWER
            self.depth = 0 if req.thinking else 1
            self.think_ct = 0
            self.ans_ct = 0
            self.truncated = False
            self.forced = deque()
            self.rng = random.Random(req.seed)
        self.finished = False

    def snapshot(self) -> GenRequest:
        """Continuation request that resumes this sequence exactly."""
        return GenRequest(
            prompt=self.req.prompt, budget=self.req.budget, seed=self.req.seed,
            thinking=self.req.thinking, max_answer_tokens=self.req.max_answer_tokens,
            meta=self.req.meta, prior_ids=list(self.gen_ids),
            state=dict(phase=self.phase, depth=self.depth, think_ct=self.think_ct,
                       ans_ct=self.ans_ct, truncated=self.truncated,
                       forced=list(self.forced), rng=self.rng.getstate()),
        )


@torch.no_grad()
def generate_batch(model, tok, requests: list[GenRequest], *, controller=None,
                   sampling: dict = THINK_SAMPLING, exempt_k: int = 10,
                   compact_every: int = 64, kv_budget_gb: float = 12.0,
                   verbose: bool = False):
    """Returns (results, continuations): results[i] is None where sequence i
    was evicted; its continuation request appears in `continuations`."""
    device = model.device
    n = len(requests)
    prefix_ids = tuple(tok(ANSWER_PREFIX, add_special_tokens=False).input_ids)
    brace_cache: dict[int, int] = {}

    def brace_delta(t: int) -> int:
        if t not in brace_cache:
            s = tok.decode([t])
            brace_cache[t] = s.count("{") - s.count("}")
        return brace_cache[t]

    # --- tokenize; manual left-padding (prompts may carry prior_ids) ---
    all_ids = []
    for r in requests:
        ids = tok(r.prompt, add_special_tokens=False).input_ids + list(r.prior_ids)
        all_ids.append(ids)
    maxlen = max(len(x) for x in all_ids)
    input_ids = torch.full((n, maxlen), PAD_ID, dtype=torch.long)
    attn = torch.zeros((n, maxlen), dtype=torch.long)
    for i, ids in enumerate(all_ids):
        input_ids[i, maxlen - len(ids):] = torch.tensor(ids)
        attn[i, maxlen - len(ids):] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)

    use_abl = controller is not None
    need_clean = use_abl and controller.use_exemption
    n_caches = 2 if need_clean else 1
    cache = DynamicCache()
    clean_cache = DynamicCache() if need_clean else None

    results: list[GenResult | None] = [None] * n
    continuations: list[GenRequest] = []
    seqs = [_Seq(r, i) for i, r in enumerate(requests)]

    def finalize(s: _Seq):
        results[s.idx] = _finalize(tok, s.req, s.gen_ids, s.think_ct, s.ans_ct,
                                   s.truncated, s.finished)

    with (controller if use_abl else nullcontext()):
        # --- prefill ---
        if need_clean:
            out_c = _forward(model, input_ids, attn, pos, clean_cache, hidden=True)
            controller.set_exempt(_clean_topk(model, out_c.hidden_states[-1], exempt_k))
            del out_c
        if use_abl:
            controller.enabled = True
            out = _forward(model, input_ids, attn, pos, cache)
            controller.enabled = False
        else:
            out = _forward(model, input_ids, attn, pos, cache)
        last_logits = out.logits[:, -1]
        del out
        pos_next = pos[:, -1] + 1

        step = 0
        while True:
            sampled = _sample_rows(last_logits, sampling, [s.rng for s in seqs])
            next_tokens = []
            for i, s in enumerate(seqs):
                if s.phase == DONE:
                    next_tokens.append(PAD_ID)
                    continue
                if s.phase == THINK and not s.forced and s.think_ct >= s.req.budget:
                    s.forced.extend(FORCE_END_THINK)
                    s.truncated = True
                was_forced = bool(s.forced)
                t = s.forced.popleft() if s.forced else sampled[i]
                s.gen_ids.append(t)
                if s.phase == THINK:
                    if t == END_THINK_ID:
                        s.phase = ANSWER
                        s.forced.extend((271,) + prefix_ids)   # "\n\n" + prefix
                    elif not was_forced:
                        s.think_ct += 1
                else:
                    if t == IM_END_ID:
                        s.phase = DONE
                        s.finished = True
                    else:
                        s.depth += brace_delta(t)
                        if not was_forced:
                            s.ans_ct += 1
                            if s.depth <= 0 or s.ans_ct >= s.req.max_answer_tokens:
                                s.phase = DONE
                next_tokens.append(t)
            if all(s.phase == DONE for s in seqs):
                break

            ids_t = torch.tensor(next_tokens, device=device).unsqueeze(1)
            attn = torch.cat([attn, torch.ones(len(seqs), 1, dtype=attn.dtype,
                                               device=device)], dim=1)
            pos_t = pos_next.unsqueeze(1)
            pos_next = pos_next + 1

            if need_clean:
                out_c = _forward(model, ids_t, attn, pos_t, clean_cache)
                controller.set_exempt(out_c.logits[:, -1:].topk(exempt_k, dim=-1).indices)
                del out_c
            if use_abl:
                controller.enabled = True
                out = _forward(model, ids_t, attn, pos_t, cache)
                controller.enabled = False
            else:
                out = _forward(model, ids_t, attn, pos_t, cache)
            last_logits = out.logits[:, -1]
            del out

            step += 1
            if step % compact_every == 0:
                keep = [i for i, s in enumerate(seqs) if s.phase != DONE]
                # KV-budget eviction: kick the longest thinkers to a later wave.
                # Straggler eviction: when a large batch has burned down to a
                # long-running tail, evict the tail so the caller can pool the
                # stragglers from many chunks into one dense later wave
                # (results are identical either way: per-sequence RNG state is
                # carried in the continuation). Only in large first-wave
                # batches (n >= 64), so shrunken resume waves can't re-evict
                # forever.
                kv_gb = len(keep) * attn.shape[1] * KV_BYTES_PER_TOKEN * n_caches / 1e9
                straggle = (n >= 64 and len(keep) <= max(2, n // 12)
                            and attn.shape[1] >= 4096)
                if (kv_gb > kv_budget_gb and len(keep) > 1) or (straggle and keep):
                    order = sorted(keep, key=lambda i: seqs[i].think_ct, reverse=True)
                    n_evict = len(keep) if straggle else max(1, len(keep) // 3)
                    for i in order[:n_evict]:
                        continuations.append(seqs[i].snapshot())
                        seqs[i].phase = DONE
                        seqs[i].evicted = True
                    keep = [i for i in keep if not getattr(seqs[i], "evicted", False)]
                    if verbose:
                        print(f"  step {step}: evicted {n_evict} at seq_len "
                              f"{attn.shape[1]}", flush=True)
                if len(keep) < 0.75 * len(seqs):
                    for i, s in enumerate(seqs):
                        if s.phase == DONE and not getattr(s, "evicted", False):
                            finalize(s)
                    if not keep:
                        return results, continuations
                    sel = torch.tensor(keep, device=device)
                    attn = attn[sel]
                    last_logits = last_logits[sel]
                    pos_next = pos_next[sel]
                    cache.batch_select_indices(sel)
                    if clean_cache is not None:
                        clean_cache.batch_select_indices(sel)
                    seqs = [seqs[i] for i in keep]
            if verbose and step % 512 == 0:
                n_alive = sum(1 for s in seqs if s.phase != DONE)
                print(f"  step {step}: {n_alive} alive, seq_len {attn.shape[1]}",
                      flush=True)

        for s in seqs:
            if not getattr(s, "evicted", False):
                finalize(s)
    return results, continuations


def _finalize(tok, req, ids, think_ct, ans_ct, truncated, finished):
    completion = tok.decode(ids, skip_special_tokens=False)
    if END_THINK_ID in ids:
        cut = ids.index(END_THINK_ID)
        answer_ids = ids[cut + 1:]
    else:
        answer_ids = [] if req.thinking else ids
    answer_text = tok.decode(answer_ids, skip_special_tokens=True)
    if not req.thinking:
        # the forced answer prefix lives in the prompt, not the completion
        answer_text = ANSWER_PREFIX + answer_text
    return GenResult(
        completion=completion, answer_text=answer_text,
        think_tokens=think_ct, answer_tokens=ans_ct,
        think_truncated=truncated, finished=finished, meta=req.meta, ids=list(ids),
    )

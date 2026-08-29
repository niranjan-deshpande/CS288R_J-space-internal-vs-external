"""Batched generation engine: thinking-budget forcing, optional J-space
ablation with clean-pass exemptions, per-sequence reproducible sampling,
batch compaction as sequences finish."""
from __future__ import annotations

import random
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from .model import END_THINK_ID, IM_END_ID, PAD_ID, THINK_SAMPLING

from .model import ANSWER_PREFIX  # noqa: E402  (re-export)

FORCE_END_THINK = (198, 151668)   # "\n</think>"
THINK, ANSWER, DONE = 0, 1, 2


@dataclass
class GenRequest:
    prompt: str                     # fully templated; ends inside <think> if thinking
    budget: int                     # max thinking tokens B
    seed: int
    thinking: bool = True
    max_answer_tokens: int = 64
    meta: dict = field(default_factory=dict)


@dataclass
class GenResult:
    completion: str
    answer_text: str
    think_tokens: int
    answer_tokens: int
    think_truncated: bool
    finished: bool
    meta: dict


def _sample_rows(logits: torch.Tensor, params: dict, rngs: list) -> list[int]:
    """Temperature + top-k + top-p filtering on GPU; final draw per row on CPU
    with that row's own RNG, so results don't depend on batch composition."""
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
    """Top-k next-token ids at every position from pre-norm hidden states."""
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


@torch.no_grad()
def generate_batch(model, tok, requests: list[GenRequest], *, controller=None,
                   sampling: dict = THINK_SAMPLING, exempt_k: int = 10,
                   compact_every: int = 64, verbose: bool = False) -> list[GenResult]:
    device = model.device
    n = len(requests)
    enc = tok([r.prompt for r in requests], return_tensors="pt", padding=True,
              add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)

    use_abl = controller is not None
    need_clean = use_abl and controller.use_exemption
    cache = DynamicCache()
    clean_cache = DynamicCache() if need_clean else None
    results: list[GenResult | None] = [None] * n

    with (controller if use_abl else nullcontext()):
        # --- prefill ---
        if need_clean:
            out_c = _forward(model, input_ids, attn, pos, clean_cache, hidden=True)
            exempt = _clean_topk(model, out_c.hidden_states[-1], exempt_k)
            del out_c
            controller.set_exempt(exempt)
        if use_abl:
            controller.enabled = True
            out = _forward(model, input_ids, attn, pos, cache)
            controller.enabled = False
        else:
            out = _forward(model, input_ids, attn, pos, cache)
        last_logits = out.logits[:, -1]
        del out

        # --- per-sequence state (indexed by current batch row) ---
        prefix_ids = tuple(tok(ANSWER_PREFIX, add_special_tokens=False).input_ids)
        brace_cache: dict[int, int] = {}

        def brace_delta(t: int) -> int:
            if t not in brace_cache:
                s = tok.decode([t])
                brace_cache[t] = s.count("{") - s.count("}")
            return brace_cache[t]

        orig = list(range(n))                  # row -> original request index
        phase = [THINK if r.thinking else ANSWER for r in requests]
        # non-thinking prompts already contain the forced "...\boxed{" prefix
        depth = [0 if r.thinking else 1 for r in requests]
        think_ct = [0] * n
        ans_ct = [0] * n
        truncated = [False] * n
        finished = [False] * n
        forced: list[deque] = [deque() for _ in range(n)]
        gen_ids: list[list[int]] = [[] for _ in range(n)]
        rngs = [random.Random(r.seed) for r in requests]
        budgets = [r.budget for r in requests]
        max_ans = [r.max_answer_tokens for r in requests]
        pos_next = pos[:, -1] + 1              # [B]

        def finalize_row(i: int):
            oi = orig[i]
            results[oi] = _finalize(tok, requests[oi], gen_ids[i], think_ct[i],
                                    ans_ct[i], truncated[i], finished[i])

        step = 0
        while True:
            sampled = _sample_rows(last_logits, sampling, rngs)
            next_tokens = []
            for i in range(len(orig)):
                if phase[i] == DONE:
                    next_tokens.append(PAD_ID)
                    continue
                # budget check happens before consuming the sample
                if phase[i] == THINK and not forced[i] and think_ct[i] >= budgets[i]:
                    forced[i].extend(FORCE_END_THINK)
                    truncated[i] = True
                was_forced = bool(forced[i])
                t = forced[i].popleft() if forced[i] else sampled[i]
                gen_ids[i].append(t)
                if phase[i] == THINK:
                    if t == END_THINK_ID:
                        phase[i] = ANSWER
                        # force the answer prefix: reasoning lives only in <think>
                        forced[i].extend((271,) + prefix_ids)  # "\n\n" + prefix
                    elif not was_forced:
                        think_ct[i] += 1
                else:  # ANSWER
                    if t == IM_END_ID:
                        phase[i] = DONE
                        finished[i] = True
                    else:
                        depth[i] += brace_delta(t)
                        if not was_forced:
                            ans_ct[i] += 1
                            if depth[i] <= 0 or ans_ct[i] >= max_ans[i]:
                                phase[i] = DONE  # boxed answer closed (or cap hit)
                next_tokens.append(t)
            if all(p == DONE for p in phase):
                break

            ids_t = torch.tensor(next_tokens, device=device).unsqueeze(1)
            attn = torch.cat([attn, torch.ones(len(orig), 1, dtype=attn.dtype,
                                               device=device)], dim=1)
            pos_t = pos_next.unsqueeze(1)
            pos_next = pos_next + 1

            if need_clean:
                out_c = _forward(model, ids_t, attn, pos_t, clean_cache)
                exempt = out_c.logits[:, -1:].topk(exempt_k, dim=-1).indices
                del out_c
                controller.set_exempt(exempt)
            if use_abl:
                controller.enabled = True
                out = _forward(model, ids_t, attn, pos_t, cache)
                controller.enabled = False
            else:
                out = _forward(model, ids_t, attn, pos_t, cache)
            last_logits = out.logits[:, -1]
            del out

            step += 1
            # --- compaction: finalize and drop finished rows ---
            if step % compact_every == 0:
                done_rows = [i for i in range(len(orig)) if phase[i] == DONE]
                if len(done_rows) > 0.25 * len(orig):
                    for i in done_rows:
                        finalize_row(i)
                    alive = [i for i in range(len(orig)) if phase[i] != DONE]
                    if not alive:
                        return results
                    sel = torch.tensor(alive, device=device)
                    attn = attn[sel]
                    last_logits = last_logits[sel]
                    pos_next = pos_next[sel]
                    cache.batch_select_indices(sel)
                    if clean_cache is not None:
                        clean_cache.batch_select_indices(sel)
                    orig = [orig[i] for i in alive]
                    phase = [phase[i] for i in alive]
                    depth = [depth[i] for i in alive]
                    think_ct = [think_ct[i] for i in alive]
                    ans_ct = [ans_ct[i] for i in alive]
                    truncated = [truncated[i] for i in alive]
                    finished = [finished[i] for i in alive]
                    forced = [forced[i] for i in alive]
                    gen_ids = [gen_ids[i] for i in alive]
                    rngs = [rngs[i] for i in alive]
                    budgets = [budgets[i] for i in alive]
                    max_ans = [max_ans[i] for i in alive]
            if verbose and step % 512 == 0:
                n_alive = sum(1 for p in phase if p != DONE)
                print(f"  step {step}: {n_alive} alive, seq_len {attn.shape[1]}",
                      flush=True)

        for i in range(len(orig)):
            finalize_row(i)
    return results


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
        think_truncated=truncated, finished=finished, meta=req.meta,
    )

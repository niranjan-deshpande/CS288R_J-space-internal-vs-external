"""Model loading and prompt construction for Qwen3-4B."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import AttentionInterface

MODEL_ID = "Qwen/Qwen3-4B"
LENS_PATH = "/workspace/artifacts/neuronpedia-jlens/qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"

THINK_ID = 151667      # <think>
END_THINK_ID = 151668  # </think>
IM_END_ID = 151645     # <|im_end|> (also eos)
PAD_ID = 151643

# Qwen3 recommended sampling settings (thinking / non-thinking modes)
THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
NOTHINK_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}

MATH_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
ANSWER_PREFIX = "Final answer: \\boxed{"


# --- GQA fast-decode attention -------------------------------------------
# transformers' sdpa integration only passes enable_gqa=True to torch sdpa
# when attention_mask is None (integrations/sdpa_attention.py::
# use_gqa_in_sdpa). With our left-padded batches the 4D mask is non-None, so
# every decode step, in every layer, repeat_kv materializes K and V to 32
# heads (4x KV read + a full 32-head write) -- the dominant decode cost at
# long sequence lengths. Passing enable_gqa together with a mask is no fix:
# torch 2.8 dispatches that combination only to the MATH kernel (slower, and
# OOMs at large batch*seq).
#
# Fix: at decode (q_len == 1) fold the GQA groups into the query-length dim,
#   q [B, 32, 1, D] -> [B, 8, 4, D],
# and call sdpa with the original 8-head K/V and the same broadcastable
# [B, 1, 1, K] bool mask. Identical reduction on the same (memory-efficient)
# kernel family -> bit-identical outputs, no KV materialization. Prefill and
# every other case falls through to the stock sdpa implementation.

ATTN_IMPL_NAME = "sdpa_gqa_decode"


def _sdpa_gqa_decode_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
):
    n_groups = getattr(module, "num_key_value_groups", 1)
    if (
        query.shape[2] == 1                       # decode: single query token
        and n_groups > 1                          # GQA model
        and attention_mask is not None            # padded batch (else stock
                                                  #  path already uses gqa)
        and attention_mask.shape[-2] == 1         # plain [B,1,1,K] mask
        and dropout == 0.0
        and kwargs.get("position_bias") is None
        and key.shape[-1] == value.shape[-1]
    ):
        bsz, n_q_heads, _, head_dim = query.shape
        # kv head i serves q heads [i*g, (i+1)*g) (repeat_kv layout), so a
        # plain reshape groups them correctly; the mask broadcasts over the
        # new "q_len" = group dim exactly as it broadcast over heads before.
        q = query.reshape(bsz, n_q_heads // n_groups, n_groups, head_dim)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, key, value,
            attn_mask=attention_mask, dropout_p=0.0, scale=scaling,
            is_causal=False,
        )
        attn = attn.reshape(bsz, n_q_heads, 1, head_dim)
        return attn.transpose(1, 2).contiguous(), None
    return sdpa_attention_forward(
        module, query, key, value, attention_mask,
        dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)


def _register_fast_decode_attention() -> str:
    """Registers the interface (idempotent) and returns the impl name to pass
    as attn_implementation. Registering the mask function too is essential: a
    custom attention name absent from ALL_MASK_ATTENTION_FUNCTIONS makes
    transformers skip mask creation entirely (masking_utils.
    _preprocess_mask_arguments), silently ignoring left padding."""
    AttentionInterface.register(ATTN_IMPL_NAME, _sdpa_gqa_decode_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTN_IMPL_NAME, sdpa_mask)
    return ATTN_IMPL_NAME
# ---------------------------------------------------------------------------


def load_model(device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device,
        attn_implementation=_register_fast_decode_attention(),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def build_prompt(tok, question: str, thinking: bool, instr: str = MATH_INSTR) -> str:
    """Chat-templated prompt. For thinking runs the <think> block is opened in
    the prompt so budget counting starts at a fixed point; for B=0 the template
    inserts an empty think block (direct answering)."""
    msgs = [{"role": "user", "content": f"{question}\n\n{instr}"}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )
    if thinking:
        text += "<think>\n"
    else:
        # direct answering (B=0): template closed an empty think block;
        # force the same answer prefix the engine forces after </think>
        text += ANSWER_PREFIX
    return text

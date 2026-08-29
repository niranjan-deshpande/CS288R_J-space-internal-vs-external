"""Model loading and prompt construction for Qwen3-4B."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def load_model(device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa",
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

"""Dissociation control: closed-book recall (paper: impaired) vs extractive QA
on the SAME facts, answer present in context (paper: intact). If ablation
breaks both, it's generic damage; if it spares extraction, it's selective
workspace removal."""
from jlens.lens import JacobianLens

from extcot.model import load_model, build_prompt, LENS_PATH, NOTHINK_SAMPLING
from extcot.engine import GenRequest, generate_batch
from extcot.ablation import AblationController

FACTS = [
    ("The capital of Australia is Canberra.", "What is the capital of Australia?", ["canberra"]),
    ("The currency of Japan is the yen.", "What is the currency of Japan?", ["yen"]),
    ("The capital of Canada is Ottawa.", "What is the capital of Canada?", ["ottawa"]),
    ("The Thames flows through London.", "Which river flows through London?", ["thames"]),
    ("Portuguese is spoken in Brazil.", "What language is spoken in Brazil?", ["portuguese"]),
    ("The capital of Nepal is Kathmandu.", "What is the capital of Nepal?", ["kathmandu"]),
    ("The currency of Switzerland is the franc.", "What is the currency of Switzerland?", ["franc"]),
    ("The capital of Kenya is Nairobi.", "What is the capital of Kenya?", ["nairobi"]),
    ("The capital of New Zealand is Wellington.", "What is the capital of New Zealand?", ["wellington"]),
    ("The currency of South Korea is the won.", "What is the currency of South Korea?", ["won"]),
    ("The capital of Egypt is Cairo.", "What is the capital of Egypt?", ["cairo"]),
    ("The capital of Peru is Lima.", "What is the capital of Peru?", ["lima"]),
    ("Mount Fuji is located in Japan.", "In which country is Mount Fuji located?", ["japan"]),
    ("The Colosseum is located in Rome.", "In which city is the Colosseum located?", ["rome"]),
    ("The capital of Spain is Madrid.", "What is the capital of Spain?", ["madrid"]),
    ("The euro is used in Spain.", "What currency is used in Spain?", ["euro"]),
    ("The capital of Russia is Moscow.", "What is the capital of Russia?", ["moscow"]),
    ("Arabic is the official language of Egypt.", "What is the official language of Egypt?", ["arabic"]),
    ("The capital of Germany is Berlin.", "What is the capital of Germany?", ["berlin"]),
    ("The pyramids of Giza are located in Egypt.", "In which country are the pyramids of Giza?", ["egypt"]),
]

INSTR = "Answer with just the answer, no explanation."


def run(model, tok, prompts_golds, controller, label):
    reqs = [GenRequest(prompt=build_prompt(tok, q, thinking=False, instr=INSTR),
                       budget=0, seed=hash((q, "ex")) % 2**31, thinking=False,
                       max_answer_tokens=24, meta={"golds": g})
            for q, g in prompts_golds]
    res, _ = generate_batch(model, tok, reqs, controller=controller,
                            sampling=NOTHINK_SAMPLING)
    ok = sum(any(g in r.answer_text.lower() for g in r.meta["golds"]) for r in res)
    print(f"[{label}] {ok}/{len(res)}", flush=True)


def main():
    model, tok = load_model()
    lens = JacobianLens.load(LENS_PATH)
    recall = [(q, g) for _, q, g in FACTS]
    extract = [(f"Context: {c}\n\nQuestion: {q}", g) for c, q, g in FACTS]
    for label, qa in [("recall", recall), ("extract", extract)]:
        run(model, tok, qa, None, f"{label} clean")
        for band, k, name in [(range(22, 32), 10, "jspace-k10"),
                              (range(22, 35), 25, "jspace-hk25"),
                              (range(22, 35), 50, "jspace-hk50")]:
            ctrl = AblationController(model, lens, band, k=k, mode="jspace")
            run(model, tok, qa, ctrl, f"{label} {name}")


if __name__ == "__main__":
    main()

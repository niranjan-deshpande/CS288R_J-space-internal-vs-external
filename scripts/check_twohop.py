"""Decisive mechanism check: J-space ablation should break two-hop latent
recall (the paper's clearest impaired capability) while random ablation and
one-hop recall stay intact."""
import torch
from jlens.lens import JacobianLens

from extcot.model import load_model, build_prompt, LENS_PATH, NOTHINK_SAMPLING
from extcot.engine import GenRequest, generate_batch
from extcot.ablation import AblationController

TWO_HOP = [
    ("What is the currency of the country shaped like a boot?", ["euro", "lira"]),
    ("What is the capital of the country where the Eiffel Tower stands?", ["paris"]),
    ("In which continent is the country whose capital is Ottawa?", ["north america"]),
    ("What language is spoken in the city famous for the Colosseum?", ["italian"]),
    ("What is the capital of the country that gifted the Statue of Liberty to the USA?", ["paris"]),
    ("Which ocean borders the country whose capital is Lima?", ["pacific"]),
    ("What is the currency of the country home to Mount Fuji?", ["yen"]),
    ("What is the capital city of the country where kangaroos are native?", ["canberra"]),
    ("Which river flows through the capital of England?", ["thames"]),
    ("What is the official language of the country whose capital is Brasilia?", ["portuguese"]),
    ("What currency is used in the country whose capital is Seoul?", ["won"]),
    ("On which continent is the largest desert that is hot?", ["africa"]),
    ("What is the capital of the country where the Taj Mahal is located?", ["delhi"]),
    ("What language is primarily spoken in the country home to the Great Barrier Reef?", ["english"]),
    ("What is the currency of the country whose flag has a red maple leaf?", ["dollar"]),
    ("Which mountain range contains the tallest peak on Earth, located in which country's border region? Name the country whose capital is Kathmandu.", ["nepal"]),
    ("What is the capital of the country famous for tulips and windmills?", ["amsterdam"]),
    ("What currency is used in the country where flamenco originated?", ["euro", "peseta"]),
    ("Which sea lies to the north of the country whose capital is Berlin?", ["baltic", "north"]),
    ("What is the national language of the country home to the pyramids of Giza?", ["arabic"]),
    ("What is the capital of the country where sushi originated?", ["tokyo"]),
    ("What currency is used in the country whose capital is Moscow?", ["ruble", "rouble"]),
    ("On which continent is the country whose capital is Nairobi?", ["africa"]),
    ("What is the capital of the country that borders both France and Portugal?", ["madrid"]),
    ("What language is spoken in the country home to the fjords near Oslo?", ["norwegian"]),
    ("What is the currency of the country where the Alps meet Lake Geneva, whose capital is Bern?", ["franc"]),
    ("Which hemisphere contains the country whose capital is Wellington?", ["southern"]),
    ("What is the capital of the country famous for maple syrup?", ["ottawa"]),
    ("What is the official currency of the country whose capital is London?", ["pound", "sterling"]),
    ("What is the capital of the country where the Nile reaches the Mediterranean?", ["cairo"]),
]

ONE_HOP = [
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Italy?", ["rome"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("What is the currency of Japan?", ["yen"]),
    ("What is the capital of Canada?", ["ottawa"]),
    ("What is the capital of Spain?", ["madrid"]),
    ("What is the currency of the United Kingdom?", ["pound", "sterling"]),
    ("What is the capital of Egypt?", ["cairo"]),
    ("What is the capital of Russia?", ["moscow"]),
    ("What is the capital of Australia?", ["canberra"]),
    ("What language is spoken in Brazil?", ["portuguese"]),
    ("What is the capital of Germany?", ["berlin"]),
    ("What is the capital of Kenya?", ["nairobi"]),
    ("What is the currency of South Korea?", ["won"]),
    ("What is the capital of Nepal?", ["kathmandu"]),
    ("What is the capital of the Netherlands?", ["amsterdam"]),
    ("What is the capital of England?", ["london"]),
    ("What is the currency of Switzerland?", ["franc"]),
    ("What is the capital of New Zealand?", ["wellington"]),
    ("What is the capital of Peru?", ["lima"]),
]

INSTR = "Answer with just the answer, no explanation."


def run(model, tok, qa, controller, label):
    reqs = [GenRequest(prompt=build_prompt(tok, q, thinking=False, instr=INSTR),
                       budget=0, seed=hash((q, "th")) % 2**31, thinking=False,
                       max_answer_tokens=24, meta={"golds": golds})
            for q, golds in qa]
    res = generate_batch(model, tok, reqs, controller=controller,
                         sampling=NOTHINK_SAMPLING)
    ok = sum(any(g in r.answer_text.lower() for g in r.meta["golds"]) for r in res)
    print(f"[{label}] {ok}/{len(res)}", flush=True)
    return res


def main():
    model, tok = load_model()
    lens = JacobianLens.load(LENS_PATH)
    conds = [
        ("clean", None),
        ("jspace 22-31", dict(band=range(22, 32), k=10, mode="jspace")),
        ("jspace 25-34", dict(band=range(25, 35), k=10, mode="jspace")),
        ("jspace 14-33", dict(band=range(14, 34), k=10, mode="jspace")),
        ("jspace 22-31 NOEX", dict(band=range(22, 32), k=10, mode="jspace", use_exemption=False)),
        ("random 22-31", dict(band=range(22, 32), k=10, mode="random", rand_seed=0)),
    ]
    for name, qa in [("two-hop", TWO_HOP), ("one-hop", ONE_HOP)]:
        for label, kw in conds:
            ctrl = AblationController(model, lens, **kw) if kw else None
            res = run(model, tok, qa, ctrl, f"{name} {label}")
        # show a few ablated answers for flavor
    ctrl = AblationController(model, lens, band=range(22, 32), k=10, mode="jspace")
    res = run(model, tok, TWO_HOP[:6], ctrl, "sample")
    for r in res:
        print("   ", repr(r.answer_text[:80]))


if __name__ == "__main__":
    main()

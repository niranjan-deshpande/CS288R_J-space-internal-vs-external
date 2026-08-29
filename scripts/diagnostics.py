"""Layer diagnostics on wikitext (paper Fig: workspace band identification).

Per layer: (a) J-lens top-1 agreement with the model's final top-1,
(b) J-lens top-1 accuracy on the actual next token, (c) excess kurtosis of
lens logits, (d) top-1 persistence across adjacent positions, (e) mean
fraction of residual norm removed by k=10 J-space ablation (exemption on).
Also logit-lens (J=I) comparison for (a).
"""
import json

import torch
from datasets import load_dataset
from jlens.lens import JacobianLens

from extcot.model import load_model, LENS_PATH


@torch.no_grad()
def main():
    model, tok = load_model()
    lens = JacobianLens.load(LENS_PATH)
    dev = model.device

    gain = model.model.norm.weight.float()
    W_tilde = (model.lm_head.weight.float() * gain[None, :]).to(torch.bfloat16)

    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in wt["text"] if len(t) > 1500][:40]

    n_layers = len(lens.source_layers)
    agree = torch.zeros(n_layers)      # lens top1 == model top1
    agree_ll = torch.zeros(n_layers)   # logit-lens top1 == model top1
    nexttok = torch.zeros(n_layers)    # lens top1 == actual next token
    kurt = torch.zeros(n_layers)
    persist = torch.zeros(n_layers)
    removal_frac = torch.zeros(n_layers)
    count = 0

    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=256).input_ids.to(dev)
        acts = {}
        hooks = []
        for li, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(
                lambda m, i, o, li=li: acts.__setitem__(li, (o[0] if isinstance(o, tuple) else o).detach())))
        out = model(input_ids=ids)
        for h in hooks:
            h.remove()
        final_logits = out.logits[0]                       # [T, V]
        model_top1 = final_logits.argmax(-1)               # [T]
        clean_top10 = final_logits.topk(10, dim=-1).indices  # [T, 10]
        target = ids[0, 1:]                                # next tokens
        T = ids.shape[1]

        for j, l in enumerate(lens.source_layers):
            h = acts[l][0].to(torch.bfloat16)              # [T, d]
            J = lens.jacobians[l].to(dev, torch.bfloat16)
            r = h @ J.T
            z = (r @ W_tilde.T).float()                    # [T, V]
            z_ll = ((h @ W_tilde.T)).float()               # logit lens (J=I)
            top1 = z.argmax(-1)
            agree[j] += (top1 == model_top1).float().mean().cpu()
            agree_ll[j] += (z_ll.argmax(-1) == model_top1).float().mean().cpu()
            nexttok[j] += (top1[:-1] == target).float().mean().cpu()
            zc = z - z.mean(-1, keepdim=True)
            k4 = (zc ** 4).mean(-1) / (zc ** 2).mean(-1) ** 2 - 3.0
            kurt[j] += k4.mean().cpu()
            persist[j] += (top1[:-1] == top1[1:]).float().mean().cpu()

            # removal fraction under the actual protocol (k=10, exemption on)
            zx = z.clone()
            zx.scatter_(-1, clean_top10, float("-inf"))
            idx = zx.topk(10, dim=-1).indices              # [T, 10]
            G = (W_tilde[idx] @ J).float()                 # [T, 10, d]
            hf = h.float()
            A = G @ G.transpose(-1, -2)
            A += torch.eye(10, device=dev) * (1e-4 * A.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1))
            b = G @ hf.unsqueeze(-1)
            coef = torch.linalg.solve(A, b)
            rem = (G.transpose(-1, -2) @ coef).squeeze(-1)
            removal_frac[j] += (rem.norm(dim=-1) / hf.norm(dim=-1)).mean().cpu()
        count += 1

    stats = {
        "layers": lens.source_layers,
        "agree_model_top1": (agree / count).tolist(),
        "agree_logitlens": (agree_ll / count).tolist(),
        "next_token_acc": (nexttok / count).tolist(),
        "excess_kurtosis": (kurt / count).tolist(),
        "top1_persistence": (persist / count).tolist(),
        "removal_frac": (removal_frac / count).tolist(),
        "n_texts": count,
    }
    with open("/workspace/jlens-cot/results/layer_diagnostics.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(f"{'L':>3} {'agree':>6} {'a_ll':>6} {'ntok':>6} {'kurt':>8} {'pers':>6} {'rem%':>6}")
    for j, l in enumerate(lens.source_layers):
        print(f"{l:>3} {stats['agree_model_top1'][j]:6.3f} {stats['agree_logitlens'][j]:6.3f} "
              f"{stats['next_token_acc'][j]:6.3f} {stats['excess_kurtosis'][j]:8.1f} "
              f"{stats['top1_persistence'][j]:6.3f} {stats['removal_frac'][j]:6.3f}")


if __name__ == "__main__":
    main()

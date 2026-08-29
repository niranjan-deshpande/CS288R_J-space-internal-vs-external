"""J-space ablation via forward hooks, following the workspace paper's protocol.

At each token position, for each layer l in the band, the residual stream
(block-l output) is projected onto the span of the top-k most active J-lens
vectors at that position and the projection is removed. The J-lens vector for
vocab token v at layer l is g_v = J_l^T (gain * w_v): the direction whose
activation the lens reads out (final RMSNorm folded in as its diagonal gain;
the positive 1/rms scalar does not affect rankings or directions).

Tokens appearing in the top-`exempt_k` of a clean forward pass at the same
position are exempt (set per step by the engine via `set_exempt`).

mode="random": norm-matched control. k fixed random orthonormal directions per
layer (seeded); the removal along them is rescaled to the norm of the J-space
removal that would have been applied at that position.
"""
from __future__ import annotations

import torch


class AblationController:
    def __init__(self, model, lens, band, *, k: int = 10, mode: str = "jspace",
                 rand_seed: int = 0, chunk: int = 32, use_exemption: bool = True):
        assert mode in ("jspace", "random")
        self.model = model
        self.k = k
        self.mode = mode
        self.band = sorted(band)
        self.chunk = chunk
        self.enabled = False
        self.use_exemption = use_exemption
        self.exempt = None  # [B, T, E] clean-top-E token ids per position
        self.stat_sum = 0.0   # running mean of ||removal|| / ||h||
        self.stat_n = 0

        dev = model.device
        gain = model.model.norm.weight.float()
        W = model.lm_head.weight.float()
        self.W_tilde = (W * gain[None, :]).to(torch.bfloat16).contiguous()  # [V, d]
        self.J = {l: lens.jacobians[l].to(dev).to(torch.bfloat16).contiguous()
                  for l in self.band}
        if mode == "random":
            gen = torch.Generator().manual_seed(rand_seed)
            self.R = {}
            for l in self.band:
                M = torch.randn(model.config.hidden_size, k, generator=gen)
                Q, _ = torch.linalg.qr(M)
                self.R[l] = Q.T.to(dev).float().contiguous()  # [k, d] orthonormal
        self._handles = []

    def set_exempt(self, ids: torch.Tensor | None):
        self.exempt = ids

    def __enter__(self):
        layers = self.model.model.layers
        for l in self.band:
            self._handles.append(layers[l].register_forward_hook(self._make_hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_hook(self, l: int):
        def fn(module, inputs, output):
            if not self.enabled:
                return None
            h = output[0] if isinstance(output, tuple) else output
            h_new = self._ablate(h, l)
            if isinstance(output, tuple):
                return (h_new,) + tuple(output[1:])
            return h_new
        return fn

    def _ablate(self, h: torch.Tensor, l: int) -> torch.Tensor:
        Bsz, T, d = h.shape
        Jl = self.J[l]
        out = torch.empty_like(h)
        eye = torch.eye(self.k, device=h.device)
        for s in range(0, T, self.chunk):
            hs = h[:, s:s + self.chunk]                       # [B, t, d]
            z = (hs @ Jl.T) @ self.W_tilde.T                  # lens logits [B, t, V]
            if self.use_exemption and self.exempt is not None:
                ex = self.exempt[:, s:s + self.chunk]
                z.scatter_(-1, ex, torch.finfo(z.dtype).min)
            idx = z.topk(self.k, dim=-1).indices              # [B, t, k]
            del z
            G = (self.W_tilde[idx] @ Jl).float()              # [B, t, k, d]
            hf = hs.float()
            A = G @ G.transpose(-1, -2)
            A = A + eye * (1e-4 * A.diagonal(dim1=-2, dim2=-1)
                           .mean(-1, keepdim=True).unsqueeze(-1))
            b = G @ hf.unsqueeze(-1)                          # [B, t, k, 1]
            coef = torch.linalg.solve(A, b)
            removal = (G.transpose(-1, -2) @ coef).squeeze(-1)  # [B, t, d]
            self.stat_sum += float((removal.norm(dim=-1) / hf.norm(dim=-1).clamp_min(1e-6)).mean())
            self.stat_n += 1
            if self.mode == "jspace":
                res = hf - removal
            else:
                R = self.R[l]                                 # [k, d] orthonormal
                pR = (hf @ R.T) @ R                           # projection onto span(R)
                nJ = removal.norm(dim=-1, keepdim=True)
                nR = pR.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                res = hf - pR * (nJ / nR)
            out[:, s:s + self.chunk] = res.to(h.dtype)
        return out

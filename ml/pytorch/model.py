"""PyTorch port of edge_node/tiny_transformer.TinyTransformerClassifier.

Same architecture, layer for layer (pre-LayerNorm, GPT-2 ordering):

    4 features -> per-feature bin embedding + slot embedding -> 5 tokens ([CLS] + 4)
              -> LayerNorm -> single-head self-attention (no biases) -> residual
              -> LayerNorm -> Linear(d, d_ff) -> ReLU -> Linear(d_ff, d) -> residual
              -> [CLS] vector -> Linear(d, 3)

With n_bins=16, d_model=8, d_ff=16 it has exactly 1,155 parameters, the same as the
hand-written model. `load_handwritten_params` copies weights across so the two can be
checked for numerical equivalence (see ml/tests/test_pytorch_model.py).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

N_FEATURES = 4
N_CLASSES = 3
FEATURE_NAMES = ["rsmi", "light_delta", "gas_index_trend", "rsmi_trend"]


class TinyTransformer(nn.Module):
    def __init__(self, n_bins: int = 16, d_model: int = 8, d_ff: int = 16):
        super().__init__()
        self.d_model = d_model
        # One embedding table per feature (not shared), like the original.
        self.emb = nn.ModuleList([nn.Embedding(n_bins, d_model) for _ in range(N_FEATURES)])
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.slot_emb = nn.Parameter(torch.empty(1, N_FEATURES + 1, d_model))
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.head = nn.Linear(d_model, N_CLASSES)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 0.1) -> None:
        """Small-normal init for weights, zero biases, unit LayerNorm gain (as in the original)."""
        for p in [*(e.weight for e in self.emb), self.cls_token, self.slot_emb,
                  self.Wq.weight, self.Wk.weight, self.Wv.weight, self.Wo.weight,
                  self.ff1.weight, self.ff2.weight, self.head.weight]:
            nn.init.normal_(p, std=std)
        for b in (self.ff1.bias, self.ff2.bias, self.head.bias):
            nn.init.zeros_(b)

    def forward(self, x_bins: torch.Tensor) -> torch.Tensor:
        """x_bins: (N, 4) int64 bin ids -> logits (N, 3)."""
        n = x_bins.shape[0]
        feat = torch.stack([self.emb[i](x_bins[:, i]) for i in range(N_FEATURES)], dim=1)
        tokens = torch.cat([self.cls_token.expand(n, -1, -1), feat], dim=1) + self.slot_emb

        h = self.ln1(tokens)
        q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(self.d_model), dim=-1)
        resid1 = tokens + self.Wo(attn @ v)

        ffn = self.ff2(torch.relu(self.ff1(self.ln2(resid1))))
        resid2 = resid1 + ffn
        return self.head(resid2[:, 0, :])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def load_handwritten_params(model: TinyTransformer, params: dict[str, np.ndarray]) -> None:
    """Copy weights from the numpy model's flat params dict. numpy uses x @ W (in, out);
    nn.Linear stores (out, in), hence the transposes."""
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=model.cls_token.dtype)  # noqa: E731
    for i, name in enumerate(FEATURE_NAMES):
        model.emb[i].weight.copy_(t(params[f"emb_{name}"]))
    model.cls_token.copy_(t(params["cls_token"]).reshape(1, 1, -1))
    model.slot_emb.copy_(t(params["slot_emb"]).unsqueeze(0))
    model.ln1.weight.copy_(t(params["ln1_gamma"])); model.ln1.bias.copy_(t(params["ln1_beta"]))
    model.ln2.weight.copy_(t(params["ln2_gamma"])); model.ln2.bias.copy_(t(params["ln2_beta"]))
    for lin, key in ((model.Wq, "Wq"), (model.Wk, "Wk"), (model.Wv, "Wv"), (model.Wo, "Wo")):
        lin.weight.copy_(t(params[key]).T)
    model.ff1.weight.copy_(t(params["W1"]).T); model.ff1.bias.copy_(t(params["b1"]))
    model.ff2.weight.copy_(t(params["W2"]).T); model.ff2.bias.copy_(t(params["b2"]))
    model.head.weight.copy_(t(params["Wc"]).T); model.head.bias.copy_(t(params["bc"]))

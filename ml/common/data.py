"""Shared data loading for every framework (hand-written, PyTorch, TensorFlow).

All frameworks must see identical inputs, so binning reuses the original
edge_node.tiny_transformer functions unchanged: quantile bin edges fitted on the
chronological-train portion only, then searchsorted into token ids. The result is
cached as .npz next to the dataset so repeated runs skip the JSON parse.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "source"))

from edge_node.tiny_transformer import (  # noqa: E402
    CLASS_NAMES, FEATURE_NAMES, compute_bin_edges, quantize,
)

LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
DEFAULT_N_BINS = 16


def load_split(dataset_path: str, n_bins: int = DEFAULT_N_BINS) -> dict:
    """Returns dict with X_train, y_train, X_test, y_test (int64), bin_edges, n_bins_actual, meta."""
    meta_path = dataset_path.replace(".jsonl", ".meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    cache_path = dataset_path.replace(".jsonl", f".bins{n_bins}.npz")

    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(dataset_path):
        z = np.load(cache_path, allow_pickle=False)
        bin_edges = {name: z[f"edges_{name}"] for name in FEATURE_NAMES}
        out = {k: z[k] for k in ("X_train", "y_train", "X_test", "y_test")}
    else:
        feats = {name: [] for name in FEATURE_NAMES}
        labels = []
        with open(dataset_path) as f:
            for line in f:
                r = json.loads(line)
                for name in FEATURE_NAMES:
                    feats[name].append(r[name])
                labels.append(LABEL_TO_IDX[r["label"]])
        cols = {name: np.asarray(v, dtype=np.float64) for name, v in feats.items()}
        y = np.asarray(labels, dtype=np.int64)
        split = meta["split_index"]
        if len(y) != meta["rows"]:
            raise ValueError(f"{dataset_path}: {len(y)} rows but meta says {meta['rows']}")
        bin_edges = {name: compute_bin_edges(cols[name][:split], n_bins) for name in FEATURE_NAMES}
        X = np.stack([quantize(cols[name], bin_edges[name]) for name in FEATURE_NAMES], axis=1)
        out = {"X_train": X[:split], "y_train": y[:split], "X_test": X[split:], "y_test": y[split:]}
        np.savez(cache_path, **out, **{f"edges_{name}": e for name, e in bin_edges.items()})

    out["bin_edges"] = bin_edges
    out["n_bins_actual"] = max(len(e) for e in bin_edges.values()) + 1
    out["meta"] = meta
    return out

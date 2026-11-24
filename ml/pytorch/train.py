#!/usr/bin/env python3
"""Train the PyTorch TinyTransformer on a dataset built by ml/data/build_datasets.py.

Mirrors scripts/train_transformer.py: same binning (16 quantile bins fitted on the
chronological-train part), Adam lr=0.05, batch 4096, 60 epochs, and keeps the weights of
the epoch with the lowest training loss. Writes to ml/artifacts/<dataset>/pytorch/:
model.pt (state_dict), bin_edges.json, metrics.json, train.log-style stdout.

Usage:
  .venv/bin/python ml/pytorch/train.py data-local/datasets/sept15.jsonl [--device cpu|mps]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from common.data import CLASS_NAMES, FEATURE_NAMES, load_split  # noqa: E402
from model import TinyTransformer, count_params  # noqa: E402


@torch.no_grad()
def accuracy(model: TinyTransformer, X: torch.Tensor, y: torch.Tensor, batch: int = 65536) -> float:
    model.eval()
    correct = 0
    for s in range(0, len(y), batch):
        correct += (model(X[s:s + batch]).argmax(dim=1) == y[s:s + batch]).sum().item()
    return correct / len(y)


@torch.no_grad()
def confusion(model: TinyTransformer, X: torch.Tensor, y: torch.Tensor) -> list[list[int]]:
    model.eval()
    pred = torch.cat([model(X[s:s + 65536]).argmax(dim=1) for s in range(0, len(y), 65536)])
    cm = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)
    for t, p in zip(y.cpu().tolist(), pred.cpu().tolist()):
        cm[t, p] += 1
    return cm.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--n-bins", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or mps (tiny model: cpu is usually as fast)")
    ap.add_argument("--out-dir", help="default: ml/artifacts/<dataset name>/pytorch")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    name = os.path.basename(args.dataset).replace(".jsonl", "")
    out_dir = args.out_dir or os.path.join(_HERE, "..", "artifacts", name, "pytorch")
    os.makedirs(out_dir, exist_ok=True)

    d = load_split(args.dataset, args.n_bins)
    dev = torch.device(args.device)
    Xtr = torch.as_tensor(d["X_train"], device=dev); ytr = torch.as_tensor(d["y_train"], device=dev)
    Xte = torch.as_tensor(d["X_test"], device=dev); yte = torch.as_tensor(d["y_test"], device=dev)
    print(f"{name}: train={len(ytr)} test={len(yte)} bins={d['n_bins_actual']} device={dev}")

    model = TinyTransformer(d["n_bins_actual"], args.d_model, args.d_ff).to(dev)
    n_params = count_params(model)
    print(f"Model: {n_params} parameters")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = torch.nn.CrossEntropyLoss()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    best_loss, best_state, history = float("inf"), None, []
    t0 = time.time()
    n = len(ytr)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(n, generator=gen).to(dev)
        total, batches = 0.0, 0
        for s in range(0, n, args.batch_size):
            idx = order[s:s + args.batch_size]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item(); batches += 1
        epoch_loss = total / batches
        history.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss, best_state = epoch_loss, copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss={epoch_loss:.4f}")
    fit_seconds = time.time() - t0
    model.load_state_dict(best_state)

    train_acc, test_acc = accuracy(model, Xtr, ytr), accuracy(model, Xte, yte)
    print(f"\nFit time: {fit_seconds:.1f}s")
    print(f"Chronological split — train accuracy: {train_acc:.4f}  test accuracy: {test_acc:.4f}")

    torch.save(model.cpu().state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "bin_edges.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in d["bin_edges"].items()}, f)
    metrics = {
        "framework": "pytorch", "torch_version": torch.__version__, "dataset": name,
        "params": n_params, "n_bins": d["n_bins_actual"], "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed, "device": str(dev),
        "fit_seconds": round(fit_seconds, 1), "best_train_loss": best_loss,
        "train_accuracy": train_acc, "test_accuracy": test_acc,
        "test_confusion": {"labels": CLASS_NAMES, "matrix": confusion(model.to(dev), Xte, yte)},
        "loss_history": history, "feature_names": FEATURE_NAMES,
        "dataset_meta": {k: d["meta"][k] for k in ("rows", "train_rows", "test_rows", "split_timestamp", "pot_a_cutoff_utc")},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

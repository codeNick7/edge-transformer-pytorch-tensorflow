#!/usr/bin/env python3
"""Train the TensorFlow/Keras TinyTransformer on a dataset built by ml/data/build_datasets.py.

Mirrors ml/pytorch/train.py and scripts/train_transformer.py: same 16 quantile bins (fitted on
the chronological-train part), Adam lr=0.05, batch 4096, 60 epochs, keep the weights of the
lowest-training-loss epoch. Writes ml/artifacts/<dataset>/tensorflow/: weights.npz (named like
the hand-written model's params), bin_edges.json, metrics.json.

Usage:
  .venv/bin/python ml/keras_tf/train.py data-local/datasets/sept15.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from common.data import CLASS_NAMES, FEATURE_NAMES, load_split  # noqa: E402
import tf_model  # noqa: E402


class KeepBestByLoss(tf.keras.callbacks.Callback):
    """Remember the weights of the epoch with the lowest training loss; restore them at the end."""

    def __init__(self):
        super().__init__()
        self.best, self.best_weights, self.history = float("inf"), None, []

    def on_epoch_end(self, epoch, logs=None):
        loss = float(logs["loss"])
        self.history.append(loss)
        if loss < self.best:
            self.best, self.best_weights = loss, [w.copy() for w in self.model.get_weights()]
        if epoch % 5 == 0 or epoch == self.params["epochs"] - 1:
            print(f"  epoch {epoch:3d}  loss={loss:.4f}")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--n-bins", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", help="default: ml/artifacts/<dataset name>/tensorflow")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    name = os.path.basename(args.dataset).replace(".jsonl", "")
    out_dir = args.out_dir or os.path.join(_HERE, "..", "artifacts", name, "tensorflow")
    os.makedirs(out_dir, exist_ok=True)

    d = load_split(args.dataset, args.n_bins)
    Xtr, ytr = d["X_train"].astype(np.int32), d["y_train"]
    Xte, yte = d["X_test"].astype(np.int32), d["y_test"]
    print(f"{name}: train={len(ytr)} test={len(yte)} bins={d['n_bins_actual']} tf={tf.__version__}")

    model = tf_model.build(d["n_bins_actual"])
    n_params = tf_model.count_params(model)
    print(f"Model: {n_params} parameters")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    keep = KeepBestByLoss()
    t0 = time.time()
    model.fit(Xtr, ytr, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, verbose=0, callbacks=[keep])
    fit_seconds = time.time() - t0

    def acc(X, y):
        pred = np.concatenate([model(X[s:s + 65536]).numpy().argmax(1) for s in range(0, len(y), 65536)])
        return float((pred == y).mean()), pred

    train_acc, _ = acc(Xtr, ytr)
    test_acc, pred = acc(Xte, yte)
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (yte, pred), 1)
    print(f"\nFit time: {fit_seconds:.1f}s")
    print(f"Chronological split — train accuracy: {train_acc:.4f}  test accuracy: {test_acc:.4f}")

    np.savez(os.path.join(out_dir, "weights.npz"), **tf_model.get_weights_dict(model))
    with open(os.path.join(out_dir, "bin_edges.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in d["bin_edges"].items()}, f)
    metrics = {
        "framework": "tensorflow", "tensorflow_version": tf.__version__, "dataset": name,
        "params": n_params, "n_bins": d["n_bins_actual"], "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed, "device": "cpu",
        "fit_seconds": round(fit_seconds, 1), "best_train_loss": keep.best,
        "train_accuracy": train_acc, "test_accuracy": test_acc,
        "test_confusion": {"labels": CLASS_NAMES, "matrix": cm.tolist()},
        "loss_history": keep.history, "feature_names": FEATURE_NAMES,
        "dataset_meta": {k: d["meta"][k] for k in ("rows", "train_rows", "test_rows", "split_timestamp", "pot_a_cutoff_utc")},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

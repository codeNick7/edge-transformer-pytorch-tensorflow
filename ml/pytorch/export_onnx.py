#!/usr/bin/env python3
"""Export a trained PyTorch TinyTransformer to ONNX and verify it with onnxruntime.

Verification: run PyTorch and onnxruntime on the first N chronological-test rows and
require identical predicted classes and max |prob diff| < 1e-5. Writes model.onnx next
to model.pt and adds an "onnx" block to metrics.json.

Usage:
  .venv/bin/python ml/pytorch/export_onnx.py data-local/datasets/sept15.jsonl [--artifacts DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from common.data import load_split  # noqa: E402
from model import TinyTransformer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--artifacts", help="default: ml/artifacts/<dataset name>/pytorch")
    ap.add_argument("--n-check", type=int, default=20000)
    args = ap.parse_args()

    name = os.path.basename(args.dataset).replace(".jsonl", "")
    art = args.artifacts or os.path.join(_HERE, "..", "artifacts", name, "pytorch")
    d = load_split(args.dataset)

    model = TinyTransformer(d["n_bins_actual"])
    model.load_state_dict(torch.load(os.path.join(art, "model.pt"), weights_only=True))
    model.eval()

    onnx_path = os.path.join(art, "model.onnx")
    example = torch.as_tensor(d["X_test"][:8])
    torch.onnx.export(
        model, (example,), onnx_path,
        input_names=["x_bins"], output_names=["logits"],
        dynamic_shapes={"x_bins": {0: torch.export.Dim("batch")}},
        dynamo=True,
    )

    X = d["X_test"][: args.n_check]
    with torch.no_grad():
        pt_prob = torch.softmax(model(torch.as_tensor(X)), dim=1).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_prob = sess.run(None, {"x_bins": X.astype(np.int64)})[0]
    ort_prob = np.exp(ort_prob - ort_prob.max(axis=1, keepdims=True))
    ort_prob /= ort_prob.sum(axis=1, keepdims=True)

    mismatches = int((pt_prob.argmax(1) != ort_prob.argmax(1)).sum())
    max_diff = float(np.abs(pt_prob - ort_prob).max())

    one = X[:1].astype(np.int64)
    for _ in range(100):
        sess.run(None, {"x_bins": one})
    t0 = time.perf_counter()
    reps = 2000
    for _ in range(reps):
        sess.run(None, {"x_bins": one})
    latency_ms = (time.perf_counter() - t0) / reps * 1000

    status = "PASS" if mismatches == 0 and max_diff < 1e-5 else "FAIL"
    print(f"ONNX check on {len(X)} test rows: {status} — {mismatches} class mismatches, max prob diff {max_diff:.2e}")
    print(f"onnxruntime single-row latency (Mac CPU): {latency_ms:.4f} ms; file size {os.path.getsize(onnx_path)} bytes")

    mpath = os.path.join(art, "metrics.json")
    metrics = json.load(open(mpath))
    metrics["onnx"] = {
        "path": "model.onnx", "bytes": os.path.getsize(onnx_path), "onnxruntime_version": ort.__version__,
        "check_rows": len(X), "class_mismatches": mismatches, "max_prob_diff": max_diff,
        "single_row_latency_ms_mac_cpu": round(latency_ms, 4), "status": status,
    }
    json.dump(metrics, open(mpath, "w"), indent=2)
    if status != "PASS":
        raise SystemExit("ONNX verification failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert the trained Keras TinyTransformer to TFLite for the UNO Q MCU and verify it.

* Input is float32 [1, 4] (bin ids as floats) so EloquentTinyML's float predict() works;
  the model casts to int32 internally (CAST op). Output: class probabilities [1, 3].
* Weights are frozen to constants first; otherwise every dense layer becomes BATCH_MATMUL,
  which tflm_cortexm 1.0.0 does not provide.
* Every op must be in ml/board_test/tflm_cortexm_ops.txt, or the script fails.
* float32: checked on the FULL chronological test set against the Keras model (TFLite reference
  kernels, no XNNPACK, closest to TFLM).
* int8 (experiment): post-training quantisation with a representative set from the train part;
  reports ops, size and full-test accuracy. Falls back to float for ops without int8 kernels.

Writes model_float32.tflite / model_int8.tflite next to weights.npz and a "tflite" block in metrics.json.

Usage:
  .venv/bin/python ml/keras_tf/convert_tflite.py data-local/datasets/sept15.jsonl
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
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ML = os.path.join(_HERE, "..")
sys.path.insert(0, _ML)
sys.path.insert(0, _HERE)

from common.data import load_split  # noqa: E402
import tf_model  # noqa: E402


def board_ops() -> set[str]:
    words = open(os.path.join(_ML, "board_test", "tflm_cortexm_ops.txt")).read().split()
    return {w[3:].replace("_", "").lower() for w in words if w.startswith("Add")}


def interpreter(tflite: bytes) -> tf.lite.Interpreter:
    it = tf.lite.Interpreter(model_content=tflite,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    return it


def op_names(it) -> list[str]:
    return sorted({o["op_name"] for o in it._get_ops_details()})


def run_all(it, X: np.ndarray) -> np.ndarray:
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    probs = np.empty((len(X), 3), np.float32)
    for i in range(len(X)):
        it.set_tensor(inp["index"], X[i:i + 1])
        it.invoke()
        probs[i] = it.get_tensor(out["index"])[0]
    return probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--artifacts", help="default: ml/artifacts/<dataset name>/tensorflow")
    ap.add_argument("--no-int8", action="store_true")
    args = ap.parse_args()

    name = os.path.basename(args.dataset).replace(".jsonl", "")
    art = args.artifacts or os.path.join(_ML, "artifacts", name, "tensorflow")
    d = load_split(args.dataset)
    model = tf_model.build(d["n_bins_actual"])
    tf_model.load_handwritten_params(model, dict(np.load(os.path.join(art, "weights.npz"))))

    fn = tf.function(lambda x: model(x, return_probs=True), input_signature=[tf.TensorSpec((1, 4), tf.float32)])
    frozen = convert_variables_to_constants_v2(fn.get_concrete_function())
    allowed = board_ops()
    Xte = d["X_test"].astype(np.float32)
    yte = d["y_test"]
    keras_probs = np.concatenate([model(Xte[s:s + 65536], return_probs=True).numpy() for s in range(0, len(yte), 65536)])
    keras_acc = float((keras_probs.argmax(1) == yte).mean())
    report = {"keras_test_accuracy": keras_acc, "test_rows": int(len(yte))}

    # ---- float32 --------------------------------------------------------------------
    f32 = tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
    it = interpreter(f32)
    ops = op_names(it)
    missing = [o for o in ops if o.replace("_", "").lower() not in allowed]
    if missing:
        raise SystemExit(f"float32 model needs ops the board lacks: {missing}")
    t0 = time.time()
    p32 = run_all(it, Xte)
    secs = time.time() - t0
    acc32 = float((p32.argmax(1) == yte).mean())
    mism = int((p32.argmax(1) != keras_probs.argmax(1)).sum())
    maxdiff = float(np.abs(p32 - keras_probs).max())
    open(os.path.join(art, "model_float32.tflite"), "wb").write(f32)
    report["float32"] = {"bytes": len(f32), "ops": ops, "test_accuracy": acc32,
                         "class_mismatches_vs_keras": mism, "max_prob_diff_vs_keras": maxdiff,
                         "mac_ref_kernel_us_per_row": round(secs / len(yte) * 1e6, 2)}
    print(f"float32: {len(f32)} bytes, ops {ops}")
    print(f"  full test set: accuracy {acc32:.4f} (Keras {keras_acc:.4f}); {mism} class mismatches vs Keras; max prob diff {maxdiff:.2e}")

    # ---- int8 experiment ----------------------------------------------------------------
    if not args.no_int8:
        rng = np.random.default_rng(0)
        rep_idx = rng.choice(len(d["y_train"]), size=2000, replace=False)
        rep = d["X_train"][rep_idx].astype(np.float32)

        def representative():
            for row in rep:
                yield [row[None, :]]

        conv = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative
        try:
            i8 = conv.convert()
            it8 = interpreter(i8)
            ops8 = op_names(it8)
            missing8 = [o for o in ops8 if o.replace("_", "").lower() not in allowed]
            p8 = run_all(it8, Xte)
            acc8 = float((p8.argmax(1) == yte).mean())
            open(os.path.join(art, "model_int8.tflite"), "wb").write(i8)
            tensor_types = sorted({str(t["dtype"].__name__) for t in it8.get_tensor_details()})
            report["int8"] = {"bytes": len(i8), "ops": ops8, "ops_missing_on_board": missing8,
                              "tensor_dtypes": tensor_types, "test_accuracy": acc8,
                              "class_mismatches_vs_float32": int((p8.argmax(1) != p32.argmax(1)).sum())}
            print(f"int8: {len(i8)} bytes, ops {ops8}, dtypes {tensor_types}")
            print(f"  full test set: accuracy {acc8:.4f}; {report['int8']['class_mismatches_vs_float32']} class mismatches vs float32;"
                  f" ops missing on board: {missing8 or 'none'}")
        except Exception as e:  # report, don't hide
            report["int8"] = {"error": f"{type(e).__name__}: {e}"[:500]}
            print(f"int8 conversion failed: {report['int8']['error']}")

    mpath = os.path.join(art, "metrics.json")
    metrics = json.load(open(mpath))
    metrics["tflite"] = report
    json.dump(metrics, open(mpath, "w"), indent=2)


if __name__ == "__main__":
    main()

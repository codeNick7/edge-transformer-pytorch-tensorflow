"""Keras port must match the hand-written transformer exactly."""

import importlib.util
import os
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

_ML = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ML, "..", "source"))

from edge_node.tiny_transformer import TinyTransformerClassifier  # noqa: E402

_spec = importlib.util.spec_from_file_location("tf_model", os.path.join(_ML, "keras_tf", "tf_model.py"))
tf_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tf_model)


def test_param_count_matches_handwritten():
    ref = TinyTransformerClassifier(n_bins=16)
    assert tf_model.count_params(tf_model.build(16)) == sum(p.size for p in ref.params.values()) == 1155


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_matches_handwritten_with_same_weights(seed):
    ref = TinyTransformerClassifier(n_bins=16, seed=seed)
    rng = np.random.default_rng(seed)
    for k in ("ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta", "b1", "b2", "bc"):
        ref.params[k] = ref.params[k] + rng.standard_normal(ref.params[k].shape) * 0.1
    model = tf_model.build(16)
    tf_model.load_handwritten_params(model, ref.params)
    X = rng.integers(0, 16, size=(512, 4))
    ref_logits, _ = ref.forward(X)
    out = model(tf.constant(X, tf.int32)).numpy()
    # Keras weights are float32 (numpy reference is float64), so compare at float32 precision.
    np.testing.assert_allclose(out, ref_logits, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(out.argmax(1), ref_logits.argmax(1))


def test_tflite_conversion_uses_only_board_ops():
    model = tf_model.build(16)
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    fn = tf.function(lambda x: model(x, return_probs=True), input_signature=[tf.TensorSpec((1, 4), tf.int32)])
    # Weights must be frozen to constants: left as variables, every dense matmul becomes
    # BATCH_MATMUL (not available on the board) instead of FULLY_CONNECTED.
    frozen = convert_variables_to_constants_v2(fn.get_concrete_function())
    tflite = tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
    interp = tf.lite.Interpreter(model_content=tflite,
                                 experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    ops = {d["op_name"] for d in interp._get_ops_details()}
    board = open(os.path.join(_ML, "board_test", "tflm_cortexm_ops.txt")).read()
    board_ops = {w[3:] for w in board.split() if w.startswith("Add")}
    norm = lambda s: s.replace("_", "").lower()  # noqa: E731  FULLY_CONNECTED vs FullyConnected
    missing = {op for op in ops if norm(op) not in {norm(b) for b in board_ops}}
    assert "BATCH_MATMUL" not in ops
    assert not missing, f"ops not available on the board: {missing}"

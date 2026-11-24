"""PyTorch port must match the hand-written transformer exactly."""

import os
import sys

import numpy as np
import pytest
import torch

_ML = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ML, "..", "source"))
sys.path.insert(0, os.path.join(_ML, "pytorch"))

from edge_node.tiny_transformer import TinyTransformerClassifier  # noqa: E402
from model import TinyTransformer, count_params, load_handwritten_params  # noqa: E402


def test_param_count_matches_handwritten():
    ref = TinyTransformerClassifier(n_bins=16)
    assert count_params(TinyTransformer(n_bins=16)) == sum(p.size for p in ref.params.values()) == 1155


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_matches_handwritten_with_same_weights(seed):
    ref = TinyTransformerClassifier(n_bins=16, seed=seed)
    # Perturb LayerNorm/bias params too, so every tensor is exercised (not just the
    # ones with random init).
    rng = np.random.default_rng(seed)
    for k in ("ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta", "b1", "b2", "bc"):
        ref.params[k] = ref.params[k] + rng.standard_normal(ref.params[k].shape) * 0.1

    model = TinyTransformer(n_bins=16).double()
    load_handwritten_params(model, ref.params)
    X = rng.integers(0, 16, size=(512, 4))

    ref_logits, _ = ref.forward(X)
    with torch.no_grad():
        pt_logits = model(torch.as_tensor(X)).numpy()
    np.testing.assert_allclose(pt_logits, ref_logits, rtol=1e-9, atol=1e-10)


def test_gradients_match_handwritten_backward():
    ref = TinyTransformerClassifier(n_bins=16, seed=3)
    rng = np.random.default_rng(3)
    X = rng.integers(0, 16, size=(256, 4))
    y = rng.integers(0, 3, size=256)
    logits, cache = ref.forward(X)
    _, dlogits = ref.cross_entropy(logits, y)
    ref_grads = ref.backward(cache, dlogits)

    model = TinyTransformer(n_bins=16).double()
    load_handwritten_params(model, ref.params)
    loss = torch.nn.functional.cross_entropy(model(torch.as_tensor(X)), torch.as_tensor(y))
    loss.backward()
    np.testing.assert_allclose(model.Wq.weight.grad.numpy().T, ref_grads["Wq"], rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(model.head.weight.grad.numpy().T, ref_grads["Wc"], rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(model.emb[0].weight.grad.numpy(), ref_grads["emb_rsmi"], rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(model.slot_emb.grad.numpy()[0], ref_grads["slot_emb"], rtol=1e-7, atol=1e-12)


def test_onnx_export_round_trip(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    torch.manual_seed(0)
    model = TinyTransformer(n_bins=16).eval()
    X = torch.randint(0, 16, (64, 4))
    path = tmp_path / "m.onnx"
    torch.onnx.export(model, (X[:8],), str(path), input_names=["x_bins"], output_names=["logits"],
                      dynamic_shapes={"x_bins": {0: torch.export.Dim("batch")}}, dynamo=True)
    out = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x_bins": X.numpy()})[0]
    with torch.no_grad():
        np.testing.assert_allclose(out, model(X).numpy(), rtol=1e-5, atol=1e-6)

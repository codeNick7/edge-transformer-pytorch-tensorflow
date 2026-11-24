"""TensorFlow/Keras port of edge_node/tiny_transformer.TinyTransformerClassifier.

Same architecture and 1,155 parameters as the hand-written and PyTorch versions, written so
the TFLite file only needs ops that tflm_cortexm 1.0.0 provides on the UNO Q MCU
(see ml/board_test/tflm_cortexm_ops.txt — it has NO BATCH_MATMUL):

  * attention matmuls are expressed as broadcast multiply + sum:
        scores[i, j] = sum_d q[i, d] * k[j, d]      -> MUL (broadcast) + SUM
        out[i, d]    = sum_j attn[i, j] * v[j, d]   -> MUL (broadcast) + SUM
  * every per-token dense layer flattens tokens to 2-D first (RESHAPE + FULLY_CONNECTED),
    so the converter cannot emit BATCH_MATMUL for it
  * LayerNorm is written out (MEAN, SQUARED_DIFFERENCE, RSQRT, ...) with the same
    population-variance formula and eps=1e-5 as the numpy model

Math is identical to the original; only the op choice differs.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

FEATURE_NAMES = ["rsmi", "light_delta", "gas_index_trend", "rsmi_trend"]
N_FEATURES = 4
N_TOKENS = N_FEATURES + 1
N_CLASSES = 3


class TinyTransformer(tf.keras.Model):
    def __init__(self, n_bins: int = 16, d_model: int = 8, d_ff: int = 16, eps: float = 1e-5, **kw):
        super().__init__(**kw)
        self.n_bins, self.d, self.d_ff, self.eps = n_bins, d_model, d_ff, eps
        init = tf.keras.initializers.RandomNormal(stddev=0.1, seed=0)
        w = lambda name, shape, i=init: self.add_weight(name=name, shape=shape, initializer=i)  # noqa: E731
        ones, zeros = tf.keras.initializers.Ones(), tf.keras.initializers.Zeros()
        self.emb = [w(f"emb_{n}", (n_bins, d_model)) for n in FEATURE_NAMES]
        self.cls_token = w("cls_token", (1, d_model))
        self.slot_emb = w("slot_emb", (N_TOKENS, d_model))
        self.ln1_gamma, self.ln1_beta = w("ln1_gamma", (d_model,), ones), w("ln1_beta", (d_model,), zeros)
        self.Wq, self.Wk = w("Wq", (d_model, d_model)), w("Wk", (d_model, d_model))
        self.Wv, self.Wo = w("Wv", (d_model, d_model)), w("Wo", (d_model, d_model))
        self.ln2_gamma, self.ln2_beta = w("ln2_gamma", (d_model,), ones), w("ln2_beta", (d_model,), zeros)
        self.W1, self.b1 = w("W1", (d_model, d_ff)), w("b1", (d_ff,), zeros)
        self.W2, self.b2 = w("W2", (d_ff, d_model)), w("b2", (d_model,), zeros)
        self.Wc, self.bc = w("Wc", (d_model, N_CLASSES)), w("bc", (N_CLASSES,), zeros)

    # -- helpers ------------------------------------------------------------------------
    def _layernorm(self, x, gamma, beta):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.math.squared_difference(x, mean), axis=-1, keepdims=True)
        return (x - mean) * tf.math.rsqrt(var + self.eps) * gamma + beta

    def _token_dense(self, x, W, b=None):
        """(N, T, d_in) @ W -> (N, T, d_out) via a 2-D matmul (FULLY_CONNECTED in TFLite)."""
        d_in, d_out = W.shape
        y = tf.matmul(tf.reshape(x, (-1, d_in)), W)
        if b is not None:
            y = y + b
        return tf.reshape(y, (-1, N_TOKENS, d_out))

    # -- forward ------------------------------------------------------------------------
    def call(self, x_bins, return_probs: bool = False):
        x_bins = tf.cast(x_bins, tf.int32)
        feat = tf.stack([tf.gather(self.emb[i], x_bins[:, i]) for i in range(N_FEATURES)], axis=1)  # (N, 4, d)
        cls = tf.zeros_like(feat[:, :1, :]) + self.cls_token  # (N, 1, d), batch-size agnostic, no TILE op
        tokens = tf.concat([cls, feat], axis=1) + self.slot_emb  # (N, 5, d)

        h = self._layernorm(tokens, self.ln1_gamma, self.ln1_beta)
        q, k, v = self._token_dense(h, self.Wq), self._token_dense(h, self.Wk), self._token_dense(h, self.Wv)
        scores = tf.reduce_sum(q[:, :, None, :] * k[:, None, :, :], axis=-1) / math.sqrt(self.d)  # (N, 5, 5)
        attn = tf.nn.softmax(scores, axis=-1)
        attn_v = tf.reduce_sum(attn[:, :, :, None] * v[:, None, :, :], axis=2)  # (N, 5, d)
        resid1 = tokens + self._token_dense(attn_v, self.Wo)

        h2 = self._layernorm(resid1, self.ln2_gamma, self.ln2_beta)
        ffn = self._token_dense(tf.nn.relu(self._token_dense(h2, self.W1, self.b1)), self.W2, self.b2)
        resid2 = resid1 + ffn

        logits = tf.matmul(resid2[:, 0, :], self.Wc) + self.bc
        return tf.nn.softmax(logits, axis=-1) if return_probs else logits


def count_params(model: TinyTransformer) -> int:
    return int(sum(np.prod(v.shape) for v in model.trainable_weights))


def build(n_bins: int = 16, d_model: int = 8, d_ff: int = 16) -> TinyTransformer:
    m = TinyTransformer(n_bins, d_model, d_ff)
    m(tf.zeros((1, N_FEATURES), tf.int32))  # create weights
    return m


def load_handwritten_params(model: TinyTransformer, params: dict[str, np.ndarray]) -> None:
    """Same (in, out) matrix convention as the numpy model, so no transposes."""
    for i, n in enumerate(FEATURE_NAMES):
        model.emb[i].assign(params[f"emb_{n}"])
    for name in ("cls_token", "slot_emb", "ln1_gamma", "ln1_beta", "Wq", "Wk", "Wv", "Wo",
                 "ln2_gamma", "ln2_beta", "W1", "b1", "W2", "b2", "Wc", "bc"):
        getattr(model, name).assign(params[name])


def get_weights_dict(model: TinyTransformer) -> dict[str, np.ndarray]:
    out = {f"emb_{n}": model.emb[i].numpy() for i, n in enumerate(FEATURE_NAMES)}
    for name in ("cls_token", "slot_emb", "ln1_gamma", "ln1_beta", "Wq", "Wk", "Wv", "Wo",
                 "ln2_gamma", "ln2_beta", "W1", "b1", "W2", "b2", "Wc", "bc"):
        out[name] = getattr(model, name).numpy()
    return out

"""Equivalence tests for the hyper-connection weight-head Pallas kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

HC = 4
MIX = (2 + HC) * HC
EPS = 1e-6
ITERS = 20


def _reference(mixed, scale, base, *, hc=HC, eps=EPS, iters=ITERS, out_dtype=jnp.bfloat16):
    """The XLA form the kernel replaces, transcribed from ``Glm5NextHyperConnection``."""
    num_tokens = mixed.shape[0]
    pre_w, post_w, comb_w = jnp.split(mixed, [hc, 2 * hc], axis=-1)
    pre_b, post_b, comb_b = jnp.split(base, [hc, 2 * hc])
    pre_s, post_s, comb_s = (scale[i] for i in range(3))

    pre = jax.nn.sigmoid(pre_w * pre_s + pre_b) + eps
    post = 2.0 * jax.nn.sigmoid(post_w * post_s + post_b)

    comb_logits = comb_w.reshape(num_tokens, hc, hc) * comb_s + comb_b.reshape(hc, hc)
    comb = jax.nn.softmax(comb_logits, axis=-1) + eps
    comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)

    def step(_, c):
        c = c / (jnp.sum(c, axis=-1, keepdims=True) + eps)
        return c / (jnp.sum(c, axis=-2, keepdims=True) + eps)

    comb = jax.lax.fori_loop(0, iters - 1, step, comb)
    return pre, post.astype(out_dtype), comb.astype(out_dtype)


def _inputs(seed, num_tokens):
    keys = jax.random.split(jax.random.key(seed), 3)
    # ``mixed`` is the projection output after the per-token RMS rescale, which
    # lands around O(1).
    mixed = jax.random.normal(keys[0], (num_tokens, MIX), dtype=jnp.float32) * 1.2
    scale = jax.random.normal(keys[1], (3,), dtype=jnp.float32) * 0.5
    base = jax.random.normal(keys[2], (MIX,), dtype=jnp.float32) * 0.3
    return mixed, scale, base


# 13 and 3000 are not multiples of any block size: they exercise the pad path.
@pytest.mark.parametrize("num_tokens", [1, 8, 13, 64, 512, 2048, 3000, 4096])
def test_matches_reference(num_tokens):
    from sgl_jax.srt.kernels.hyper_connection import hc_weights_pallas

    mixed, scale, base = _inputs(num_tokens, num_tokens)
    want = _reference(mixed, scale, base)
    got = hc_weights_pallas(mixed, scale, base, hc=HC, eps=EPS, iters=ITERS, interpret=True)

    for name, a, b in zip(("pre", "post", "comb"), want, got):
        assert a.shape == b.shape, name
        assert a.dtype == b.dtype, name
        # The kernel does the same fp32 arithmetic in the same order, so the
        # only slack is one bf16 ulp on the cast outputs.
        np.testing.assert_allclose(
            np.asarray(b, np.float32),
            np.asarray(a, np.float32),
            rtol=0,
            atol=2e-3,
            err_msg=name,
        )


def test_comb_is_doubly_stochastic():
    from sgl_jax.srt.kernels.hyper_connection import hc_weights_pallas

    mixed, scale, base = _inputs(7, 256)
    _, _, comb = hc_weights_pallas(
        mixed, scale, base, hc=HC, eps=EPS, iters=ITERS, out_dtype=jnp.float32, interpret=True
    )
    comb = np.asarray(comb, np.float64)
    np.testing.assert_allclose(comb.sum(-1), 1.0, atol=1e-4)
    np.testing.assert_allclose(comb.sum(-2), 1.0, atol=1e-4)


def test_rejects_bad_shapes():
    from sgl_jax.srt.kernels.hyper_connection import hc_weights_pallas

    mixed, scale, base = _inputs(0, 32)
    with pytest.raises(ValueError, match="base must be"):
        hc_weights_pallas(mixed, scale, base[:-1], hc=HC, eps=EPS, iters=ITERS, interpret=True)
    with pytest.raises(ValueError, match="scale must be"):
        hc_weights_pallas(mixed, scale[:2], base, hc=HC, eps=EPS, iters=ITERS, interpret=True)
    with pytest.raises(ValueError, match="mixed must be"):
        hc_weights_pallas(mixed[:, :-1], scale, base, hc=HC, eps=EPS, iters=ITERS, interpret=True)

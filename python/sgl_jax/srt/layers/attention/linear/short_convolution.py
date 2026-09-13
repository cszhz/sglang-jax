"""Short depthwise causal conv1d used by linear-attention backends (e.g. KDA).

The convolution is intentionally implemented as a stateless function (not an
nnx Module) so backends can freely combine it with their own weight containers
and cache layouts. Two execution paths are provided:

* ``decode`` — single-token step that appends the new token to a per-sequence
  ``[B, D, K-1]`` cache, runs the conv on the resulting K-token window, and
  drops the oldest slot before writing back.
* ``extend`` — variable-length packed prefill that consumes ``cu_seqlens``,
  running the conv as shifted elementwise passes over the packed tokens and
  patching the few per-sequence boundary tokens that read the prior cache.

State convention follows vLLM: cache has width ``K-1`` and stores the
prior ``K-1`` tokens (the current token is supplied via ``x`` at call time).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.profiling_utils import named_scope

# Map of supported activation names → callable. ``None`` means identity.
_ACTIVATION_FNS: dict[str | None, Callable[[jax.Array], jax.Array] | None] = {
    None: None,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "tanh": jnp.tanh,
}


def _resolve_activation(
    activation: str | Callable[[jax.Array], jax.Array] | None,
) -> Callable[[jax.Array], jax.Array] | None:
    """Resolve an activation spec to a callable (or None for identity).

    Accepts either a name from ``_ACTIVATION_FNS`` or a user-supplied callable.
    """
    if activation is None or callable(activation):
        return activation
    if activation not in _ACTIVATION_FNS:
        raise ValueError(
            f"short_convolution activation must be one of {sorted(k for k in _ACTIVATION_FNS if k is not None)} "
            f"or a callable; got {activation!r}"
        )
    return _ACTIVATION_FNS[activation]


def short_convolution(
    x: jax.Array,
    weight: jax.Array,
    cache: jax.Array,
    cu_seqlens: jax.Array | None,
    forward_mode: ForwardMode,
    bias: jax.Array | None = None,
    activation: str | Callable[[jax.Array], jax.Array] | None = "silu",
) -> tuple[jax.Array, jax.Array]:
    """Depthwise causal conv1d with per-sequence cache.

    Args:
        x: ``[T, D]`` for ``EXTEND`` (packed varlen) or ``[B, D]`` for ``DECODE``.
        weight: depthwise kernel ``[D, K]``.
        cache: per-sequence rolling buffer ``[B, D, K-1]`` storing the prior
            ``K-1`` tokens (zeros for fresh sequences). The current token is
            supplied via ``x`` and not written into the input cache.
        cu_seqlens: ``[N+1]`` cumulative sequence lengths; required for
            ``EXTEND``, ignored for ``DECODE``.
        forward_mode: ``ForwardMode.DECODE`` or ``ForwardMode.EXTEND``.
        bias: optional ``[D]`` channel bias added before the activation.
        activation: name (e.g. ``"silu"``, ``"gelu"``, ``"sigmoid"``), a
            user-supplied callable, or ``None`` for identity.

    Returns:
        ``(y, new_cache)`` where ``y`` matches the leading dims of ``x`` and
        ``new_cache`` has the same shape as ``cache``.
    """
    activation_fn = _resolve_activation(activation)

    weight = _normalize_weight(weight)

    if forward_mode == ForwardMode.DECODE:
        return _decode_conv(x, weight, cache, bias, activation_fn)
    if cu_seqlens is None:
        raise ValueError("short_convolution(EXTEND) requires cu_seqlens")
    return _extend_conv(x, weight, cache, cu_seqlens, bias, activation_fn)


def _normalize_weight(weight: jax.Array) -> jax.Array:
    """Reduce common conv-weight layouts to ``[D, K]``."""
    # Squeeze the depthwise singleton axis if the loader handed us [D, 1, K].
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight[:, 0, :]
    return weight


def _apply_activation(
    y: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array] | None,
) -> jax.Array:
    if activation_fn is None:
        return y
    return activation_fn(y)


@named_scope("short_conv_decode")
def _decode_conv(
    x: jax.Array,  # [B, D]
    conv_kernel: jax.Array,  # [D, K]
    cache: jax.Array,  # [B, D, K-1]
    bias: jax.Array | None,
    activation_fn: Callable[[jax.Array], jax.Array] | None,
) -> tuple[jax.Array, jax.Array]:
    # expand x shape from [B, D] to [B, D, 1]
    new_cache = jnp.concatenate([cache, x[..., None]], axis=-1)
    y = jnp.einsum("bck,ck->bc", new_cache, conv_kernel.astype(new_cache.dtype))
    if bias is not None:
        y = y + bias.astype(y.dtype)
    y = _apply_activation(y, activation_fn)
    # return the last K-1 conv state
    return y, new_cache[:, :, 1:]


@named_scope("short_conv_extend")
def _extend_conv(
    x: jax.Array,  # [T, D]
    conv_kernel: jax.Array,  # [D, K]
    cache: jax.Array,  # [B, D, K-1]
    cu_seqlens: jax.Array,
    bias: jax.Array | None,
    activation_fn: Callable[[jax.Array], jax.Array] | None,
) -> tuple[jax.Array, jax.Array]:
    T, D = x.shape
    K = conv_kernel.shape[-1]
    W = K - 1  # cache width
    N = cu_seqlens.shape[0] - 1
    idx_dtype = cu_seqlens.dtype

    # A causal depthwise conv over packed tokens is K shifted copies of x
    # scaled per channel. Padding x with W zero rows turns tap j into the plain
    # slice ``xp[j : j+T]``, so the whole convolution is K fused multiply-adds
    # on [T, D] -- no [T, D, K] window and no transpose of a 4-element axis
    # into the minor position. Accumulating in fp32 also lands closer to an
    # fp32 reference than the bf16 einsum this replaces (2.2e-3 vs 5.2e-3).
    xp = jnp.concatenate([jnp.zeros((W, D), x.dtype), x], axis=0)
    acc = jnp.zeros((T, D), jnp.float32)
    for j in range(K):
        acc = acc + conv_kernel[:, j].astype(jnp.float32) * xp[j : j + T].astype(jnp.float32)

    # The shifts are wrong exactly where a window reaches back past a sequence
    # start -- the first W tokens of each sequence, at most N*W rows out of T,
    # which are the only tokens that ever read the cache. Recompute those
    # exactly and scatter them back; everything here is [N, W, ...], so the
    # cost does not scale with T.
    starts = cu_seqlens[:-1]
    ends = cu_seqlens[1:]
    head = starts[:, None] + jnp.arange(W, dtype=idx_dtype)[None, :]  # [N, W]
    head_valid = head < ends[:, None]
    safe_head = jnp.clip(head, 0, jnp.maximum(T - 1, 0))

    offsets = jnp.arange(K, dtype=idx_dtype) - (K - 1)
    source_idx = safe_head[:, :, None] + offsets[None, None, :]  # [N, W, K]
    from_x = source_idx >= starts[:, None, None]
    safe_x_idx = jnp.clip(source_idx, 0, jnp.maximum(T - 1, 0))
    # x[safe_x_idx]: [N, W, K, D] (advanced indexing puts the index axes first).
    x_window = jnp.swapaxes(x[safe_x_idx], 2, 3).astype(jnp.float32)  # [N, W, D, K]

    # cache holds the prior W = K-1 tokens at slots [0, W-1]. Map source
    # position p (where p < starts[seq]) to cache slot ``W + (p - starts)``.
    cache_pos = jnp.clip(W + source_idx - starts[:, None, None], 0, W - 1)
    cache_window = jnp.take_along_axis(
        jnp.broadcast_to(cache[:, None], (N, W, D, W)),  # [N, W, D, W]
        cache_pos[:, :, None, :],  # [N, W, 1, K] -> broadcasts over D
        axis=3,
    ).astype(
        jnp.float32
    )  # [N, W, D, K]
    window = jnp.where(from_x[:, :, None, :], x_window, cache_window)
    patch = jnp.einsum("nidk,dk->nid", window, conv_kernel.astype(jnp.float32))

    # Slots past a short sequence's end are routed to a scratch row rather than
    # clipped onto a real token, where a duplicate index could clobber a
    # genuine patch.
    acc = (
        jnp.concatenate([acc, jnp.zeros((1, D), acc.dtype)], axis=0)
        .at[jnp.where(head_valid, head, T)]
        .set(patch)[:T]
    )

    if bias is not None:
        acc = acc + bias.astype(acc.dtype)
    y = _apply_activation(acc, activation_fn).astype(x.dtype)

    # Compute the new per-sequence cache: the last W = K-1 input tokens of
    # each sequence, falling back to the prior cache when the sequence is
    # shorter than W.
    state_offsets = jnp.arange(W, dtype=idx_dtype)
    final_idx = ends[:, None] - W + state_offsets[None, :]
    final_from_x = final_idx >= starts[:, None]
    safe_final_idx = jnp.clip(final_idx, 0, jnp.maximum(T - 1, 0))
    final_x = jnp.swapaxes(x[safe_final_idx], 1, 2)
    final_cache_pos = jnp.clip(W + final_idx - starts[:, None], 0, W - 1)
    final_cache = jnp.take_along_axis(cache, final_cache_pos[:, None, :], axis=2)
    new_cache = jnp.where(final_from_x[:, None, :], final_x, final_cache)

    return y, new_cache


__all__ = ["short_convolution"]

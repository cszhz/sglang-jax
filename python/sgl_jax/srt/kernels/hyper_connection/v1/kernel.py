"""Hyper-connection weight head for TPU.

Turns the ``[T, (2+hc)*hc]`` projection output of one mHC site into the three
weight tensors ``pre``/``post``/``comb``. The arithmetic is the reference
arithmetic; the point of the kernel is the op count.

The XLA version of this tail is ~80 tiny ops per site — a sigmoid pair, a
softmax, and ``2*iters - 1`` reduce/divide pairs of Sinkhorn — every one of
them on a ``[T, 4, 4]`` tensor. At decode ``T`` is the batch size, so each op
moves a few hundred bytes and is pure launch overhead: a v7x core dispatches
one in ~5.8 us regardless of size. A 45-layer forward runs 90 sites, which is
~7000 ops and the dominant term in decode latency. Folding the whole tail into
one Pallas call makes it 1 op per site, and the Sinkhorn iterations become
unrolled VPU arithmetic inside the kernel.

Tokens occupy the TPU lane dimension in the VMEM compute layout, so the two
Sinkhorn reductions (both over ``hc``) run across sublanes and never need a
cross-lane permutation. Prefill sees little benefit — there the same ops are
large enough to be bandwidth-bound — but it is not slower either.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

SAFE_AUTO_BT = 2048


def get_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def _auto_block_tokens(num_tokens: int) -> int:
    """Smallest power of two that covers ``num_tokens``, capped at ``SAFE_AUTO_BT``.

    Everything here is ``[mix, BT]``-shaped, so VMEM is ~24 * BT * 4 bytes for
    the input plus a comb working set of the same order: a few hundred KB at
    BT=2048. The cap keeps the grid from degenerating into one giant block on
    long prefills; it is not a pressure limit. Powers of two rather than exact
    divisors, because the alternative — the largest divisor of a ragged token
    count — is 1 for a prime batch size, and a grid of T single-token steps
    would be far worse than the XLA path this replaces. Ragged counts get
    padded instead; the real shapes here (bucketed decode batches, the chunked
    prefill size) are powers of two and pad to nothing.
    """
    bt = 8
    while bt < num_tokens and bt < SAFE_AUTO_BT:
        bt *= 2
    return bt


def _hc_weights_kernel(
    mixed_ref,  # [BT, mix] f32
    scale_ref,  # [3, 1] f32
    base_ref,  # [mix, 1] f32
    pre_ref,  # [BT, hc] f32
    post_ref,  # [BT, hc] out_dtype
    comb_ref,  # [BT, hc*hc] out_dtype
    *,
    hc: int,
    eps: float,
    iters: int,
):
    block_tokens = mixed_ref.shape[0]

    # Transpose once: tokens into lanes. Every reduction below is then over
    # sublanes, which the VPU does without a permute.
    mix_t = mixed_ref[...].astype(jnp.float32).T  # [mix, BT]
    scale = scale_ref[...]
    base = base_ref[...]

    # The 3 scales apply to the pre / post / comb output groups. Doing the
    # scale-and-bias per group, rather than expanding scale to [mix, 1] in the
    # caller, keeps the expansion off the XLA graph: 4 more ops per site times
    # 90 sites is exactly the kind of thing this kernel exists to delete.
    # ``scale[i:i+1]`` is [1, 1] and broadcasts over both axes.
    pre = jax.nn.sigmoid(mix_t[:hc] * scale[0:1] + base[:hc]) + eps
    post = 2.0 * jax.nn.sigmoid(mix_t[hc : 2 * hc] * scale[1:2] + base[hc : 2 * hc])

    # [hc*hc, BT] -> [h, k, BT]. The reference indexes ``comb[t, h, k]`` and
    # flattens h-major, so h is the outer sublane group. Its ``axis=-1`` (over
    # k) is axis 1 here, and its ``axis=-2`` (over h) is axis 0.
    comb_logits = mix_t[2 * hc :] * scale[2:3] + base[2 * hc :]
    comb = comb_logits.reshape(hc, hc, block_tokens)
    comb = jax.nn.softmax(comb, axis=1) + eps
    comb = comb / (jnp.sum(comb, axis=0, keepdims=True) + eps)

    # Sinkhorn-Knopp: one column normalization above, then ``iters - 1``
    # row+column pairs. Unrolled as a Python loop — ``iters`` is static and the
    # body is a handful of VPU instructions, so 19 copies are cheap to compile
    # and emit zero XLA ops. The eps is added to the *sum*, as in the
    # reference, not to the quotient.
    for _ in range(iters - 1):
        comb = comb / (jnp.sum(comb, axis=1, keepdims=True) + eps)
        comb = comb / (jnp.sum(comb, axis=0, keepdims=True) + eps)

    # Transpose back inside the kernel rather than returning ``[*, BT]`` and
    # letting the caller do ``.T``: that would be 3 more XLA ops per site.
    pre_ref[...] = pre.T
    post_ref[...] = post.T.astype(post_ref.dtype)
    comb_ref[...] = comb.reshape(hc * hc, block_tokens).T.astype(comb_ref.dtype)


def hc_weights_pallas(
    mixed: jax.Array,  # [T, (2+hc)*hc] f32
    scale: jax.Array,  # [3] f32
    base: jax.Array,  # [(2+hc)*hc] f32
    *,
    hc: int,
    eps: float,
    iters: int,
    out_dtype: jnp.dtype = jnp.bfloat16,
    block_tokens: int | None = None,
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(pre [T, hc] f32, post [T, hc], comb [T, hc, hc])``.

    ``pre`` stays f32 because its consumer is the fp32 ``collapsed`` einsum;
    ``post`` and ``comb`` come out in ``out_dtype``, which is where the caller
    would have cast them anyway.
    """
    mix = (2 + hc) * hc
    num_tokens = mixed.shape[0]
    if mixed.shape != (num_tokens, mix):
        raise ValueError(f"mixed must be [T, {mix}], got {mixed.shape}.")
    if base.shape != (mix,):
        raise ValueError(f"base must be [{mix}], got {base.shape}.")
    if scale.shape != (3,):
        raise ValueError(f"scale must be [3], got {scale.shape}.")
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}.")

    bt = block_tokens or _auto_block_tokens(num_tokens)
    # Zero-pad a ragged token count up to a whole number of blocks. Zeros are
    # finite, so the pad rows produce finite garbage that we slice back off;
    # nothing downstream sees them. Costs one pad op plus three slices, and
    # only on shapes that are not a multiple of ``bt``.
    padded = -(-num_tokens // bt) * bt
    if padded != num_tokens:
        mixed = jnp.pad(mixed, ((0, padded - num_tokens), (0, 0)))

    # Column-shaped so the kernel can broadcast them over the token lanes. Both
    # are plain reshapes of a parameter, which XLA folds into the layout — no
    # op lands in the graph.
    scale_col = scale.reshape(3, 1)
    base_col = base.reshape(mix, 1)

    pre, post, comb = pl.pallas_call(
        functools.partial(_hc_weights_kernel, hc=hc, eps=eps, iters=iters),
        grid=(padded // bt,),
        in_specs=[
            pl.BlockSpec((bt, mix), lambda i: (i, 0)),
            pl.BlockSpec((3, 1), lambda i: (0, 0)),
            pl.BlockSpec((mix, 1), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((bt, hc), lambda i: (i, 0)),
            pl.BlockSpec((bt, hc), lambda i: (i, 0)),
            pl.BlockSpec((bt, hc * hc), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded, hc), jnp.float32),
            jax.ShapeDtypeStruct((padded, hc), out_dtype),
            jax.ShapeDtypeStruct((padded, hc * hc), out_dtype),
        ],
        interpret=get_interpret() if interpret is None else interpret,
        name="hc_weights",
    )(mixed.astype(jnp.float32), scale_col, base_col)

    if padded != num_tokens:
        pre = pre[:num_tokens]
        post = post[:num_tokens]
        comb = comb[:num_tokens]

    # The comb reshape splits the minor axis of a contiguous array: free.
    return pre, post, comb.reshape(num_tokens, hc, hc)

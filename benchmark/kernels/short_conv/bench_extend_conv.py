"""Compare formulations of the varlen prefill path in ``short_convolution``.

``_extend_conv`` builds the K-tap causal window explicitly: it gathers
``x[source_idx]`` into ``[T, K, D]``, transposes to ``[T, D, K]``, gathers a
matching ``[T, D, K]`` from the per-sequence cache, selects between them, and
only then contracts the ``K`` axis against the depthwise kernel. At the
GLM-5.3-Flash prefill shape (T=16384 tokens per chunk, D=512 channels per
TP shard, K=4) each of those intermediates is 67 MB of bf16, so a convolution
whose inputs and outputs total 34 MB moves an order of magnitude more than
that -- and the transpose puts a 4-element axis into the minor position, which
is the worst layout TPU has.

In the 200K prefill profile the three lines cost 6.8% of device time
(``:158`` 4.28%, ``:164`` 1.39%, ``:170`` 1.15%) across 102 call sites per
forward (34 KDA layers x q/k/v).

Two alternatives:

``per_tap``   -- keep the same gathers but do them one tap at a time, so the
                 widest live buffer is ``[T, D]`` instead of ``[T, D, K]``.
                 Small, local change.

``shift``     -- observe that a causal depthwise conv over packed tokens is
                 just ``K`` shifted copies of ``x`` scaled by per-channel
                 weights, and that the cache is only ever read by the first
                 ``K-1`` tokens of each sequence. So compute the whole thing
                 as fused elementwise work on ``[T, D]`` and patch the at most
                 ``N*(K-1)`` boundary tokens afterwards. No ``[T, D, K]`` ever
                 exists and no transpose is needed.

Both must reproduce ``current`` bit-for-bit-ish; the harness checks that.

    python bench_extend_conv.py --seq-lens 16384 --channels 512
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

# --------------------------------------------------------------------------
# Variants. Signatures match _extend_conv minus bias/activation plumbing,
# which is identical across all three and not what is being measured.
# --------------------------------------------------------------------------


def current(x, w, cache, cu_seqlens):
    """Verbatim ``short_convolution._extend_conv`` window construction."""
    T = x.shape[0]
    K = w.shape[-1]
    W = K - 1

    token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    starts = cu_seqlens[:-1][seq_ids]

    offsets = jnp.arange(K, dtype=cu_seqlens.dtype) - (K - 1)
    source_idx = token_idx[:, None] + offsets[None, :]
    from_x = source_idx >= starts[:, None]

    safe_x_idx = jnp.clip(source_idx, 0, jnp.maximum(T - 1, 0))
    x_window = jnp.swapaxes(x[safe_x_idx], 1, 2)  # [T, D, K]

    cache_pos = jnp.clip(W + source_idx - starts[:, None], 0, W - 1)
    cache_window = jnp.take_along_axis(cache[seq_ids], cache_pos[:, None, :], axis=2)
    window = jnp.where(from_x[:, None, :], x_window, cache_window)
    return jnp.einsum("tck,ck->tc", window, w.astype(window.dtype))


def per_tap(x, w, cache, cu_seqlens):
    """Same gathers, one tap at a time: widest live buffer is ``[T, D]``."""
    T = x.shape[0]
    K = w.shape[-1]
    W = K - 1

    token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    starts = cu_seqlens[:-1][seq_ids]

    acc = jnp.zeros(x.shape, dtype=jnp.float32)
    for j in range(K):
        source = token_idx - (K - 1 - j)
        sx = jnp.clip(source, 0, jnp.maximum(T - 1, 0))
        cpos = jnp.clip(W + source - starts, 0, W - 1)
        src = jnp.where((source >= starts)[:, None], x[sx], cache[seq_ids, :, cpos])
        acc = acc + w[:, j].astype(jnp.float32) * src.astype(jnp.float32)
    return acc.astype(x.dtype)


def shift(x, w, cache, cu_seqlens):
    """Shifted elementwise sum over ``[T, D]`` plus a sparse boundary patch.

    Tap ``j`` reads source position ``t - (K-1-j)``. Padding ``x`` with ``K-1``
    zero rows turns every tap into the slice ``xp[j : j+T]``, so the whole
    convolution is ``K`` fused multiply-adds on ``[T, D]``. That is wrong only
    where a window reaches back past a sequence start -- the first ``K-1``
    tokens of each sequence, at most ``N*(K-1)`` rows out of ``T`` -- which are
    recomputed exactly and scattered back.
    """
    T, D = x.shape
    K = w.shape[-1]
    W = K - 1
    N = cu_seqlens.shape[0] - 1
    idt = cu_seqlens.dtype

    xp = jnp.concatenate([jnp.zeros((W, D), x.dtype), x], axis=0)
    acc = jnp.zeros((T, D), jnp.float32)
    for j in range(K):
        acc = acc + w[:, j].astype(jnp.float32) * xp[j : j + T].astype(jnp.float32)
    y = acc.astype(x.dtype)

    # --- boundary patch: tokens [starts[n], starts[n]+W) of every sequence ---
    starts = cu_seqlens[:-1]
    ends = cu_seqlens[1:]
    head = starts[:, None] + jnp.arange(W, dtype=idt)[None, :]  # [N, W]
    valid = head < ends[:, None]
    ht = jnp.clip(head, 0, jnp.maximum(T - 1, 0))

    offsets = jnp.arange(K, dtype=idt) - (K - 1)
    src = ht[:, :, None] + offsets[None, None, :]  # [N, W, K]
    from_x = src >= starts[:, None, None]
    sx = jnp.clip(src, 0, jnp.maximum(T - 1, 0))
    xw = jnp.swapaxes(x[sx], 2, 3)  # [N, W, D, K]
    cpos = jnp.clip(W + src - starts[:, None, None], 0, W - 1)
    cw = jnp.take_along_axis(
        jnp.broadcast_to(cache[:, None], (N, W, D, W)), cpos[:, :, None, :], axis=3
    )
    patch = jnp.einsum("nidk,dk->nid", jnp.where(from_x[:, :, None, :], xw, cw), w.astype(x.dtype))

    # Invalid slots are routed to a scratch row rather than clipped onto a real
    # token, where a duplicate index could clobber a genuine patch.
    dest = jnp.where(valid, head, T)
    return jnp.concatenate([y, jnp.zeros((1, D), y.dtype)], axis=0).at[dest].set(patch)[:T]


def shift_nopatch(x, w, cache, cu_seqlens):
    """``shift`` without the boundary patch -- wrong, but shows its cost."""
    T, D = x.shape
    K = w.shape[-1]
    xp = jnp.concatenate([jnp.zeros((K - 1, D), x.dtype), x], axis=0)
    acc = jnp.zeros((T, D), jnp.float32)
    for j in range(K):
        acc = acc + w[:, j].astype(jnp.float32) * xp[j : j + T].astype(jnp.float32)
    return acc.astype(x.dtype)


def reference_f32(x, w, cache, cu_seqlens):
    """Exact fp32 recomputation, to judge which variant is actually closer."""
    T = x.shape[0]
    K = w.shape[-1]
    W = K - 1
    token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    starts = cu_seqlens[:-1][seq_ids]
    offsets = jnp.arange(K, dtype=cu_seqlens.dtype) - (K - 1)
    source_idx = token_idx[:, None] + offsets[None, :]
    from_x = source_idx >= starts[:, None]
    sx = jnp.clip(source_idx, 0, jnp.maximum(T - 1, 0))
    xw = jnp.swapaxes(x[sx], 1, 2).astype(jnp.float32)
    cpos = jnp.clip(W + source_idx - starts[:, None], 0, W - 1)
    cw = jnp.take_along_axis(cache[seq_ids], cpos[:, None, :], axis=2).astype(jnp.float32)
    window = jnp.where(from_x[:, None, :], xw, cw)
    return jnp.einsum("tck,ck->tc", window, w.astype(jnp.float32))


# --------------------------------------------------------------------------


def time_fn(fn, args, warmup, iters):
    jitted = jax.jit(fn)
    out = jax.block_until_ready(jitted(*args))
    for _ in range(warmup):
        out = jitted(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(jitted(*args))
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples)), out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 16384])
    p.add_argument("--channels", type=int, default=512, help="D per TP shard")
    p.add_argument("--kernel", type=int, default=4)
    p.add_argument("--n-req", type=int, nargs="+", default=[1, 4])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    D, K, W = args.channels, args.kernel, args.kernel - 1
    print(f"short_conv EXTEND on {jax.devices()[0].device_kind}: D={D} K={K}")

    for T in args.seq_lens:
        for n_req in args.n_req:
            bounds = np.linspace(0, T, n_req + 1).astype(np.int32)
            cu = jnp.asarray(bounds)
            x = jnp.asarray(rng.standard_normal((T, D)) * 0.1, dtype=jnp.bfloat16)
            w = jnp.asarray(rng.standard_normal((D, K)) * 0.5, dtype=jnp.bfloat16)
            cache = jnp.asarray(rng.standard_normal((n_req, D, W)) * 0.1, jnp.bfloat16)

            # Bytes a perfect implementation must move: read x, write y.
            ideal = 2 * T * D * 2
            print(
                f"\nT={T} n_req={n_req}  (x = {T*D*2/1e6:.0f} MB, "
                f"[T,D,K] intermediate = {T*D*K*2/1e6:.0f} MB)"
            )
            exact = np.asarray(jax.jit(reference_f32)(x, w, cache, cu), np.float32)
            denom = max(float(np.max(np.abs(exact))), 1e-6)
            variants = (
                ("current", current),
                ("per_tap", per_tap),
                ("shift", shift),
                ("shift_nopatch", shift_nopatch),
            )
            for name, fn in variants:
                ms, out = time_fn(fn, (x, w, cache, cu), args.warmup, args.iters)
                err = float(np.max(np.abs(np.asarray(out, np.float32) - exact))) / denom
                print(
                    f"  {name:14s} {ms:8.3f} ms   "
                    f"{ideal/(ms/1e3)/1e12:5.2f} TB/s of ideal traffic   "
                    f"err_vs_f32={err:.2e}"
                )


if __name__ == "__main__":
    main()

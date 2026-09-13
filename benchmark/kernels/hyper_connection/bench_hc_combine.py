"""Compare formulations of the mHC scatter-back in ``_hc_combine``.

``glm5_next._hc_combine`` writes the stream mixer as

    mixed = jnp.einsum("thk,thd->tkd", comb, residual_streams)
    return post[:, :, None] * block_out[:, None, :] + mixed

which XLA lowers to a batched dot with ``M = K = hc = 4`` against ``N = D``.
That shape uses 4 of the MXU's 128 rows and, being a dot, cannot fuse into the
elementwise add that consumes it -- so the ``[T, hc, D]`` result makes a full
round trip through HBM. ``hc`` is a compile-time 4, so the same arithmetic can
be written as an unrolled elementwise sum, which is a pure loop fusion and
should let the add ride along for free.

At 200K context this site is ~6% of prefill device time (90 sites per forward),
so it is worth knowing which lowering actually wins rather than assuming.

    python bench_hc_combine.py --seq-lens 16384 --hidden 4096
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np


def combine_einsum(comb, residual, post, block_out):
    """Current formulation: batched dot_general, then a separate add."""
    mixed = jnp.einsum("thk,thd->tkd", comb, residual)
    return post[:, :, None] * block_out[:, None, :] + mixed


def combine_unrolled(comb, residual, post, block_out):
    """``hc`` is static, so spell the contraction out as elementwise work."""
    hc = comb.shape[1]
    mixed = post[:, :, None] * block_out[:, None, :]
    for h in range(hc):
        mixed = mixed + comb[:, h, :, None] * residual[:, h, None, :]
    return mixed


def collapse_upcast(pre, streams):
    """Current formulation at ``glm5_next.py:193``: upcast then contract."""
    return jnp.einsum("th,thd->td", pre, streams.astype(jnp.float32)).astype(streams.dtype)


def collapse_preferred(pre, streams):
    """Feed the MXU the bf16 operands it already had; accumulate in fp32."""
    return jnp.einsum("th,thd->td", pre, streams, preferred_element_type=jnp.float32).astype(
        streams.dtype
    )


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
    return float(np.median(samples)), jitted(*args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 16384])
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--hc", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    print(f"mHC on {jax.devices()[0].device_kind}: hc={args.hc} D={args.hidden}")

    for seq_len in args.seq_lens:

        def bf16(*shape):
            return jnp.asarray(rng.standard_normal(shape), dtype=jnp.bfloat16)

        comb = bf16(seq_len, args.hc, args.hc)
        residual = bf16(seq_len, args.hc, args.hidden)
        post = bf16(seq_len, args.hc)
        block_out = bf16(seq_len, args.hidden)
        pre = bf16(seq_len, args.hc)

        print(f"\nT={seq_len}  (residual streams = " f"{residual.size * 2 / 1e6:.0f} MB)")
        variants = [
            ("_hc_combine einsum", combine_einsum, (comb, residual, post, block_out)),
            ("_hc_combine unrolled", combine_unrolled, (comb, residual, post, block_out)),
            ("collapse upcast-f32", collapse_upcast, (pre, residual)),
            ("collapse preferred-f32", collapse_preferred, (pre, residual)),
        ]
        ref = {}
        for name, fn, fn_args in variants:
            ms, out = time_fn(fn, fn_args, args.warmup, args.iters)
            group = name.split()[0]
            note = ""
            if group in ref:
                a = np.asarray(out, np.float32)
                b = np.asarray(ref[group], np.float32)
                denom = max(float(np.max(np.abs(b))), 1e-6)
                note = f"  rel_diff={float(np.max(np.abs(a - b))) / denom:.2e}"
            else:
                ref[group] = out
            print(f"  {name:24s} {ms:8.3f} ms{note}")


if __name__ == "__main__":
    main()

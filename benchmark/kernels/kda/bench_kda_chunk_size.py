"""Sweep the KDA chunked-prefill kernel's ``chunk_size``.

``chunk_kda_fwd`` takes ``chunk_size`` as a static argument and every caller
leaves it at the default 64 -- there is no tuning table for it, unlike MLA
(``kernels/mla/v2/tuned_block_sizes.py``) or fused_moe
(``kernels/fused_moe/v2/tuned_block_configs.py``). ``chunk_size`` is the BT of
all four kernel stages, so it sets the intra-chunk solve cost (O(BT^2) per
chunk, i.e. O(T*BT) overall) against the number of sequential inter-chunk
recurrence steps (T/BT). The optimum is hardware- and shape-dependent and is
worth finding: on a 200K-context GLM-5.3-Flash prefill, ``kda_fwd_intra`` alone
is ~25% of device time.

This benchmarks one shard, i.e. the shapes a single core actually sees under
``shard_map`` in ``KDABackend._forward_extend`` -- pass ``--num-heads`` already
divided by the tensor-parallel degree.

Example (GLM-5.3-Flash, TP=16 -> 64/16 = 4 heads per shard, 16384-token chunked
prefill):

    python bench_kda_chunk_size.py --seq-len 16384 --num-heads 4 --head-dim 128
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.kda import chunk_kda


def build_inputs(seq_len, num_heads, head_dim, value_dim, num_seqs, seed=0):
    """Random inputs in the layout ``_forward_extend`` hands the kernel."""
    rng = np.random.default_rng(seed)

    def rn(*shape, dtype=jnp.bfloat16):
        return jnp.asarray(rng.standard_normal(shape), dtype=dtype)

    def unit(*shape):
        """Unit-norm rows along the last axis.

        The intra-chunk delta rule inverts ``I + tril(beta * K K^T)`` by a
        Neumann series, which only converges for bounded ``|k|``. Real KDA k/q
        come out of an RMSNorm, so raw standard normals at K=128 diverge to NaN
        and make the cross-chunk_size output comparison useless.
        """
        x = rng.standard_normal(shape)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return jnp.asarray(x, dtype=jnp.bfloat16)

    # Packed varlen layout: B == 1, sequences concatenated along T.
    q = unit(1, seq_len, num_heads, head_dim)
    k = unit(1, seq_len, num_heads, head_dim)
    v = rn(1, seq_len, num_heads, value_dim)
    g = rn(1, seq_len, num_heads, head_dim, dtype=jnp.float32)
    beta = jnp.asarray(rng.random((1, seq_len, num_heads)), dtype=jnp.bfloat16)

    # Even split; the kernel only reads cu_seqlens, not the token contents.
    bounds = np.linspace(0, seq_len, num_seqs + 1).astype(np.int32)
    cu_seqlens = jnp.asarray(bounds)

    initial_state = jnp.zeros((num_seqs, num_heads, head_dim, value_dim), jnp.float32)
    A_log = jnp.asarray(rng.standard_normal(num_heads), jnp.float32)
    dt_bias = jnp.asarray(rng.standard_normal((num_heads, head_dim)), jnp.float32)
    return q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias


def time_chunk_size(inputs, chunk_size, sub_chunk_size, scale, lower_bound, warmup, iters):
    q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias = inputs

    @jax.jit
    def run(q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias):
        o, final_state, *_ = chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            chunk_size=chunk_size,
            sub_chunk_size=sub_chunk_size,
        )
        return o, final_state

    out = jax.block_until_ready(run(*inputs))
    for _ in range(warmup):
        out = run(*inputs)
    jax.block_until_ready(out)

    # Median of per-iteration timings: a single stray host hiccup should not
    # decide which chunk_size wins.
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(run(*inputs))
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples)), out[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, default=16384, help="packed token count")
    p.add_argument("--num-heads", type=int, default=4, help="heads PER SHARD")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--value-dim", type=int, default=None, help="defaults to head-dim")
    p.add_argument("--num-seqs", type=int, default=1)
    p.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
        help="powers of two that divide --seq-len",
    )
    p.add_argument(
        "--sub-chunk-sizes",
        nargs="+",
        default=["none"],
        help="secondary-chunking BS per chunk_size; 'none' keeps the "
        "all-elementwise path. Must divide chunk_size.",
    )
    p.add_argument("--gate-lower-bound", type=float, default=-5.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    value_dim = args.value_dim if args.value_dim is not None else args.head_dim
    scale = args.head_dim**-0.5

    inputs = build_inputs(args.seq_len, args.num_heads, args.head_dim, value_dim, args.num_seqs)

    print(
        f"KDA chunk_size sweep on {jax.devices()[0].device_kind}: "
        f"T={args.seq_len} H={args.num_heads} K={args.head_dim} V={value_dim} "
        f"N={args.num_seqs}"
    )
    subs = [None if x == "none" else int(x) for x in args.sub_chunk_sizes]
    print(f"{'chunk_size':>10} {'BS':>6} {'ms':>9} {'vs BT=64':>9}")

    results = {}
    ref = None
    for cs in args.chunk_sizes:
        if cs & (cs - 1):
            print(f"{cs:>10}   skipped (not a power of two)")
            continue
        if args.seq_len % cs:
            print(f"{cs:>10}   skipped (seq_len not divisible)")
            continue
        for bs in subs:
            if bs is not None and cs % bs:
                continue
            tag = "none" if bs is None else str(bs)
            try:
                ms, o = time_chunk_size(
                    inputs, cs, bs, scale, args.gate_lower_bound, args.warmup, args.iters
                )
            except Exception as e:  # a tiling the kernel cannot fit
                print(f"{cs:>10} {tag:>6}   failed: {type(e).__name__}: {str(e)[:80]}")
                continue
            results[(cs, bs)] = ms
            if ref is None:
                ref = o
            else:
                # Same math, different tiling -- outputs must agree.
                a = np.asarray(o, np.float32)
                b = np.asarray(ref, np.float32)
                if not (np.isfinite(a).all() and np.isfinite(b).all()):
                    print(f"{cs:>10}   WARNING: non-finite output, correctness check void")
                else:
                    diff = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-6))
                    if diff > 0.05:
                        print(f"{cs:>10} {tag:>6}   WARNING: differs from first (rel {diff:.3g})")
            base = results.get((64, None))
            rel = f"{base / ms:.2f}x" if base else "-"
            print(f"{cs:>10} {tag:>6} {ms:9.3f} {rel:>9}")

    if results:
        best = min(results, key=results.get)
        base = results.get((64, None))
        gain = f"  ({base / results[best]:.2f}x vs BT=64/BS=none)" if base else ""
        print(
            f"\nbest: chunk_size={best[0]} sub_chunk_size={best[1]} "
            f"at {results[best]:.3f} ms{gain}"
        )


if __name__ == "__main__":
    main()

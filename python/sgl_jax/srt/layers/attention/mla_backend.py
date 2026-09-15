"""Attention backend for the absorbed MLA path (MLA v2 Pallas kernel).

This backend wraps `mla_ragged_paged_attention` with a 4D paged latent KV cache
(see `MLATokenToKVPool`). It follows the same `(q, k, v, layer, forward_batch,
pool, **kwargs)` contract as the MHA FlashAttention backend so that
`RadixAttention` remains the unified entry point. Callers pass the absorbed
latent Q as `q`, the latent `c_kv` as both `k` and `v`, and the rope parts as
the `q_rope` / `k_rope` kwargs; the backend reassembles the 4-tuple
`(ql_nope, q_pe, new_kv_c, new_k_pe)` the Pallas kernel consumes. The output
is the latent attention `o_latent: [T, n_h, kv_lora_rank]`; the caller
projects it through `W_UV → W_O` (see `docs/design/MLA.md` §3.9).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.mla.v2.kernel import cdiv, mla_ragged_paged_attention
from sgl_jax.srt.kernels.mla.v2.kv_write import write_new_kv
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.utils.profiling_utils import named_scope

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.mem_cache.memory_pool import KVCache
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@register_pytree_node_class
@dataclass
class MLAAttentionMetadata:
    """Per-forward metadata for the MLA v2 kernel.

    Drops `custom_mask` / `swa_page_indices` / `attention_sink` (RFC §3.8). The
    `cu_kv_lens` field carries the page-aligned KV cumulative lengths used by
    the kernel to locate each sequence's pages in the ragged `page_indices`
    layout (matches RPA v3 addressing).
    """

    cu_q_lens: jax.Array = None
    cu_kv_lens: jax.Array = None
    page_indices: jax.Array = None
    seq_lens: jax.Array = None
    distribution: jax.Array = None

    def tree_flatten(self):
        children = (
            self.cu_q_lens,
            self.cu_kv_lens,
            self.page_indices,
            self.seq_lens,
            self.distribution,
        )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.cu_q_lens = children[0]
        obj.cu_kv_lens = children[1]
        obj.page_indices = children[2]
        obj.seq_lens = children[3]
        obj.distribution = children[4]
        return obj


@dataclass
class MLAAttentionBackend(AttentionBackend):
    """Absorbed-MLA attention backend backed by the v2 Pallas kernel."""

    def __init__(
        self,
        num_attn_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        page_size: int = 1,
        mesh: jax.sharding.Mesh = None,
        attention_data_partition_axis: str = "data",
        vmem_limit_bytes: int | None = None,
        # Default block params to None so the kernel's auto-tuned-table
        # lookup path triggers (see kernel.py mla_ragged_paged_attention
        # entry). Callers (e.g. accuracy tests) can still pass explicit
        # tuples to bypass lookup.
        num_kv_pages_per_block: tuple[int, int, int] | None = None,
        num_queries_per_block: tuple[int, int, int] | None = None,
        # decode_batch_size: kernel-internal microbatch for the BATCHED_DECODE
        # branch. The v2 kernel runs `floor(num_decode_seqs / decode_batch_size)
        # * decode_batch_size` requests through a batched path (q_len=1, q-tile
        # of size `decode_batch_size`) to keep the MXU busy on MLA's MQA-on-
        # latent decode (single q row per seq otherwise underutilizes the q
        # axis); the remainder falls back to per-seq decode.
        # When num_kv_pages_per_block is None, the kernel's lookup path also
        # overrides this with the tuned value if the "decode" entry hits.
        decode_batch_size: int = 4,
    ):
        assert page_size > 1, (
            "MLA attention backend does not support page_size=1. "
            "The MLA v2 kernel packs KV slots and infers the effective page size "
            "from cache_kv.shape[1] * kv_packing, which makes page_size=1 "
            "inconsistent with allocator and metadata page semantics."
        )
        self.num_heads = num_attn_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
        self.mesh = mesh
        self.attention_data_partition_axis = attention_data_partition_axis
        if vmem_limit_bytes is None:
            from jax.experimental.pallas import tpu as pltpu

            vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
        self.vmem_limit_bytes = vmem_limit_bytes
        self.num_kv_pages_per_block = num_kv_pages_per_block
        self.num_queries_per_block = num_queries_per_block
        self.decode_batch_size = decode_batch_size
        self.tp_size = mesh.shape["tensor"] if mesh is not None else 1
        # Smallest *padded* decode batch at which the request split is worth it;
        # 0 disables it. The split's cost per step is flat in the batch size --
        # every rank does one request's worth of full-head attention whether the
        # batch is 2 or 16 -- so below the crossover it is a pure loss. See
        # `_decode_split_range`. The padded batch is what the graph is compiled
        # for (`--precompile-bs-paddings`), so this stays a trace-time decision.
        self.decode_batch_split = int(
            os.environ.get("SGLANG_JAX_MLA_DECODE_BATCH_SPLIT", "0") or 0
        )

        self.forward_metadata = nnx.data(MLAAttentionMetadata())

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Build per-batch metadata, DP-aware.

        Mirrors `FlashAttention.get_forward_metadata`: reshapes all per-request
        arrays to `(dp_size, per_dp_bs_size)` and computes cumsums per DP rank
        so each rank's metadata is independent.
        """
        metadata = MLAAttentionMetadata()

        if batch.dp_size <= 0:
            raise ValueError(f"Invalid dp_size: {batch.dp_size}")
        if batch.per_dp_bs_size <= 0:
            raise ValueError(f"Invalid per_dp_bs_size: {batch.per_dp_bs_size}")
        if batch.per_dp_bs_size * batch.dp_size != len(batch.seq_lens):
            raise ValueError(
                "Inconsistent DP batch metadata: expected per_dp_bs_size * dp_size == len(seq_lens), "
                f"got {batch.per_dp_bs_size} * {batch.dp_size} != {len(batch.seq_lens)}"
            )
        if len(batch.cache_loc) % batch.dp_size != 0:
            raise ValueError(
                "Inconsistent cache_loc layout for DP sharding: "
                f"len(cache_loc)={len(batch.cache_loc)} is not divisible by dp_size={batch.dp_size}"
            )

        total_loc_len = len(batch.cache_loc)
        per_dp_loc_len = total_loc_len // batch.dp_size

        cache_loc_2d = batch.cache_loc.reshape(batch.dp_size, per_dp_loc_len)
        strided_2d = cache_loc_2d[:, :: self.page_size]
        page_indices = (strided_2d // self.page_size).ravel()

        if batch.forward_mode == ForwardMode.EXTEND:
            ext_2d = batch.extend_seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
            cu_q_2d = np.zeros((batch.dp_size, batch.per_dp_bs_size + 1), dtype=np.int32)
            cu_q_2d[:, 1:] = np.cumsum(ext_2d, axis=1)
            cu_q_lens = cu_q_2d.ravel()
        elif batch.forward_mode == ForwardMode.DECODE:
            single_cu = np.arange(batch.per_dp_bs_size + 1, dtype=np.int32)
            cu_q_lens = np.tile(single_cu, batch.dp_size)
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        seq_lens = batch.seq_lens
        aligned_seq_lens = (
            (batch.seq_lens + self.page_size - 1) // self.page_size
        ) * self.page_size

        aligned_2d = aligned_seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
        cu_kv_2d = np.zeros((batch.dp_size, batch.per_dp_bs_size + 1), dtype=np.int32)
        cu_kv_2d[:, 1:] = np.cumsum(aligned_2d, axis=1)
        cu_kv_lens = cu_kv_2d.ravel()

        seq_lens_2d = batch.seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
        local_num_seqs = np.sum(seq_lens_2d > 0, axis=1, dtype=np.int32)
        if batch.forward_mode == ForwardMode.DECODE:
            distribution = np.repeat(local_num_seqs, 3)
        elif batch.forward_mode == ForwardMode.EXTEND:
            distribution = np.column_stack(
                [np.zeros_like(local_num_seqs), np.zeros_like(local_num_seqs), local_num_seqs]
            ).ravel()
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        (
            metadata.cu_q_lens,
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (cu_q_lens, cu_kv_lens, page_indices, seq_lens, distribution),
            sharding=(NamedSharding(self.mesh, P(self.attention_data_partition_axis))),
        )
        return metadata

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {
            "num_attn_heads": self.num_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "page_size": self.page_size,
            "mesh": self.mesh,
            "attention_data_partition_axis": self.attention_data_partition_axis,
            "vmem_limit_bytes": self.vmem_limit_bytes,
            "num_kv_pages_per_block": self.num_kv_pages_per_block,
            "num_queries_per_block": self.num_queries_per_block,
            "decode_batch_size": self.decode_batch_size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(
            num_attn_heads=aux_data["num_attn_heads"],
            kv_lora_rank=aux_data["kv_lora_rank"],
            qk_nope_head_dim=aux_data["qk_nope_head_dim"],
            qk_rope_head_dim=aux_data["qk_rope_head_dim"],
            v_head_dim=aux_data["v_head_dim"],
            page_size=aux_data["page_size"],
            mesh=aux_data.get("mesh"),
            attention_data_partition_axis=aux_data.get("attention_data_partition_axis", "data"),
            vmem_limit_bytes=aux_data["vmem_limit_bytes"],
            num_kv_pages_per_block=aux_data["num_kv_pages_per_block"],
            num_queries_per_block=aux_data["num_queries_per_block"],
            decode_batch_size=aux_data["decode_batch_size"],
        )
        obj.forward_metadata = children[0]
        return obj

    def _decode_split_range(self, num_decode_seqs: jax.Array, num_tokens: int):
        """Give each tensor rank a contiguous slice of the decode batch.

        The MLA latent cache is replicated across all `tp_size` ranks (see
        `MLATokenToKVPool.kv_partition_axis`), so the usual head split leaves
        every rank streaming the *whole* cache to serve its 1/tp of the heads.
        At 200K context and 12 concurrent requests that is 2.46 GB per layer
        per rank -- 16x more traffic than the work requires, and it dominates
        decode (68% of step time, measured).

        Splitting by request instead inverts it: rank r computes *all* heads
        for its own requests and touches only their pages. Same FLOPs, 1/tp
        the bytes per rank, and no LSE needed -- each output row is produced
        whole by exactly one rank, so plain `psum_scatter` puts it back.

        The cost is load balance: with `num_decode_seqs < tp_size` some ranks
        idle, and the step is as slow as the rank holding the longest request.
        Splitting by page instead would balance perfectly but needs the kernel
        to emit an LSE. Measured on v7x at 200K/c12: request split 5.19x,
        page split 5.61x -- 92% of the win for none of the kernel surgery.

        Returns `(lo, hi, owner)` where `owner[t]` is the rank that produces
        output row `t`. Rows past the batch belong to rank 0, which leaves them
        holding the aliased `ql_nope` exactly as the unsplit path does.
        """
        tp = self.tp_size
        rank = jax.lax.axis_index("tensor")
        base, extra = num_decode_seqs // tp, num_decode_seqs % tp
        # Ranks [0, extra) take base+1 requests, the rest take base.
        lo = rank * base + jnp.minimum(rank, extra)
        hi = lo + base + (rank < extra)

        t = jnp.arange(num_tokens)
        big = extra * (base + 1)
        owner = jnp.where(
            t < big,
            t // (base + 1),
            # base == 0 means every request already fell in the branch above;
            # clamp only so the dead branch cannot divide by zero.
            extra + (t - big) // jnp.maximum(base, 1),
        )
        return lo, hi, jnp.where(t < num_decode_seqs, owner, 0)

    @named_scope
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        **kwargs,
    ):
        """Absorbed-MLA forward, called through ``RadixAttention``.

        Signature mirrors the other attention backends so ``RadixAttention``
        can serve as the unified entry point. The MLA-specific tensors land in
        standard slots:

            q:         [T, n_h, kv_lora_rank]            — absorbed latent query (``ql_nope``)
            k:         [T, 1, kv_lora_rank]              — latent KV (c_kv); v is the same tensor
            v:         [T, 1, kv_lora_rank]              — same object as ``k`` (MLA is MQA on the latent)
            q_rope=…:  [T, n_h, qk_rope_head_dim]        — ``q_pe``
            k_rope=…:  [T, 1, qk_rope_head_dim]          — ``k_pe`` (shared across heads)

        Internally reassembles the 4-tuple ``(ql_nope, q_pe, new_kv_c, new_k_pe)``
        the v2 Pallas kernel consumes. Returns ``(o_latent, updated_cache)``
        with ``o_latent: [T, n_h, kv_lora_rank]`` — the caller is responsible
        for projecting via ``W_UV → W_O``.

        The kernel handles ``qk_rope_head_dim=64 → 128`` and other
        ``align_to(*, 128)`` padding internally, so callers pass logical dims.
        """
        # v is the same latent tensor as k in MLA-MQA — we only need one.
        del v
        q_rope = kwargs.get("q_rope")
        k_rope = kwargs.get("k_rope")
        # NoPE models (GLM-5.3) omit both: there is no rope tail, so the kernel
        # runs a pure-latent QK and the cache holds only align_to(kv_lora_rank, 128).
        has_pe = q_rope is not None
        if has_pe != (k_rope is not None):
            raise ValueError(
                "MLAAttentionBackend requires q_rope/k_rope kwargs (q_pe/k_pe) to be "
                "passed together, or both omitted for NoPE models."
            )

        # Strip the (single) KV-head axis from the latent K/K-rope tensors.
        # Squeezing a length-1 axis preserves any caller-provided sharding on
        # the remaining dims (replicated stays replicated).
        new_kv_c = k if k.ndim == 2 else jnp.squeeze(k, axis=1)
        dpa = self.attention_data_partition_axis
        ql_nope = q
        if has_pe:
            new_k_pe = k_rope if k_rope.ndim == 2 else jnp.squeeze(k_rope, axis=1)
            new_k_pe = jax.sharding.reshard(new_k_pe, P(dpa, None))
            q_pe = q_rope
        # Request-split decode: only for a pure-decode batch on a replicated
        # cache. EXTEND / target-verify carry a mixed distribution the split
        # path deliberately drops.
        # NoPE only: taking the write out of the kernel means reproducing it,
        # and `write_new_kv` lands just the latent part -- a rope tail would
        # need the fused layout's second half written too.
        # NB: the threshold is applied in ForwardBatch.init_new, against
        # `real_bs`, NOT here against a shape. Two earlier attempts were both
        # no-ops that silently split everything: `ql_nope.shape[0]` is padded
        # to a multiple of tp by SP, and `batch_size` is pinned to the single
        # decode bucket 2*ep=32 by fused MoE. Neither carries concurrency.
        split = (
            self.decode_batch_split > 0
            and forward_batch.mla_decode_split
            and self.tp_size > 1
            and not has_pe
            and forward_batch.forward_mode == ForwardMode.DECODE
        )
        if self.decode_batch_split > 0 and forward_batch.forward_mode == ForwardMode.DECODE:
            # Once per trace, not per step -- this is the only way to see which
            # way the gate went without guessing from a TPOT number.
            logger.info(
                "MLA decode split: %s (flag=%s padded_bs=%d thresh=%d tokens=%d pe=%s)",
                "ON" if split else "off",
                forward_batch.mla_decode_split,
                forward_batch.batch_size,
                self.decode_batch_split,
                ql_nope.shape[0],
                has_pe,
            )
        if split:
            # Every rank needs all the heads of its own requests, so undo the
            # head split on the way in. 786 KB/layer at c12 -- ~4 us.
            ql_nope = jax.sharding.reshard(ql_nope, P(dpa, None, None))
            if has_pe:
                q_pe = jax.sharding.reshard(q_pe, P(dpa, None, None))

        cache = token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)
        sm_scale = (
            (1.0 / jnp.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim))
            if (layer is None or layer.scaling is None)
            else layer.scaling
        )
        sliding_window = layer.sliding_window_size if layer is not None else None
        soft_cap = layer.logit_cap if layer is not None else None

        # On the split path q arrives whole and the output is reduce-scattered
        # back to the usual head shard inside `_run`.
        q_spec = P(dpa, None, None) if split else P(dpa, "tensor", None)
        pe_specs = (
            (
                q_spec,  # q_pe       [T, n_h/tp, r]
                P(dpa, None),  # new_k_pe   [T, r]    (single latent)
            )
            if has_pe
            else ()
        )
        in_specs = (
            q_spec,  # ql_nope    [T, n_h/tp, lkv]
            P(dpa, None),  # new_kv_c   [T, lkv]  (single latent, no head axis)
            *pe_specs,
            P(dpa, None, None, None),  # cache (page axis sharded by data)
            P(dpa),  # seq_lens
            P(dpa),  # page_indices
            P(dpa),  # cu_q_lens
            P(dpa),  # cu_kv_lens
            P(dpa),  # distribution
        )
        out_specs = (
            P(dpa, "tensor", None),  # o_latent       [T, n_h/tp, lkv]
            P(dpa, None, None, None),  # updated cache  4D
        )

        def _run(
            ql_nope_,
            new_kv_c_,
            *rest,
        ):
            if has_pe:
                q_pe_, new_k_pe_, *rest = rest
            else:
                q_pe_ = new_k_pe_ = None
            (
                cache_,
                seq_lens_,
                page_indices_,
                cu_q_lens_,
                cu_kv_lens_,
                distribution_,
            ) = rest
            seq_range = owner = None
            if split:
                lo, hi, owner = self._decode_split_range(
                    distribution_[0], ql_nope_.shape[0]
                )
                seq_range = (lo, hi)
            o_, cache_ = mla_ragged_paged_attention(
                ql_nope_,
                q_pe_,
                new_kv_c_,
                new_k_pe_,
                cache_,
                seq_lens_,
                page_indices_,
                cu_q_lens_,
                cu_kv_lens_,
                distribution_,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                num_kv_pages_per_block=self.num_kv_pages_per_block,
                num_queries_per_block=self.num_queries_per_block,
                decode_batch_size=self.decode_batch_size,
                vmem_limit_bytes=self.vmem_limit_bytes,
                decode_seq_range=seq_range,
                write_kv_cache=not split,
            )
            if not split:
                return o_, cache_
            # The kernel only touched rows [lo, hi); the rest still hold the
            # aliased `ql_nope`. Zero those so the reduce-scatter sees exactly
            # one contributor per row, then hand each rank back its head shard.
            rank = jax.lax.axis_index("tensor")
            o_ = jnp.where((owner == rank)[:, None, None], o_, 0)
            o_ = jax.lax.psum_scatter(
                o_, "tensor", scatter_dimension=1, tiled=True
            )
            # The kernel read the new KV straight from `new_kv_c` this step, so
            # landing it in the cache afterwards keeps the unsplit ordering. It
            # has to happen on every rank: each holds a full replica, and next
            # step's request-to-rank assignment is not guaranteed to be the same.
            cache_ = write_new_kv(
                new_kv_c_,
                cache_,
                seq_lens_,
                page_indices_,
                cu_kv_lens_,
                distribution_[0],
            )
            return o_, cache_

        o_latent, updated_cache = jax.shard_map(
            _run,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            ql_nope,
            new_kv_c,
            *((q_pe, new_k_pe) if has_pe else ()),
            cache,
            self.forward_metadata.seq_lens,
            self.forward_metadata.page_indices,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
        )

        return o_latent, updated_cache

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        # Same scalar-prefetch budget heuristic as FlashAttention.get_max_running_reqests:
        # caps the number of concurrent requests by the per-request page slot count
        # so metadata arrays fit in TPU scalar-prefetch memory.
        num_page_per_req = cdiv(max_context_len, page_size)
        res = 1024 * 1024 // 2 // num_page_per_req // 4
        assert (
            res > 0
        ), f"max running requests: {res} must larger than 0, please increase page size or decrease max context length"
        return res

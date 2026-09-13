"""Tuned ``(chunk_size, sub_chunk_size)`` for the KDA chunked-prefill kernel.

``chunk_kda_fwd`` takes both as static arguments and every caller used to leave
``chunk_size`` (BT) at the default 64 with no sub-chunking at all. BT trades
the intra-chunk delta-rule work -- O(BT^2) per chunk, so O(T*BT) overall --
against the number of sequential inter-chunk recurrence steps, T/BT.
``sub_chunk_size`` (BS) then splits the intra-chunk score construction so the
off-diagonal tiles become MXU matmuls instead of an elementwise contraction
over a ``[BT, BT, K]`` decay tensor; that also shrinks the intermediate to
``[BS, BS, K]``, which is what lets BT go past 128 in VMEM at all. The balance
depends on the per-shard head geometry, so it is a lookup like
``kernels/mla/v2/tuned_block_sizes.py`` and
``kernels/fused_moe/v2/tuned_block_configs.py``.

Key: ``(num_heads, head_dim, value_dim)``, where ``num_heads`` is PER SHARD,
i.e. already divided by the tensor-parallel degree -- the kernel runs inside
``shard_map`` in ``KDABackend._forward_extend``.

A miss returns ``(64, None)``, the historical default, so an untuned shape
behaves exactly as before. Benchmark with
``benchmark/kernels/kda/bench_kda_chunk_size.py``.
"""

from __future__ import annotations

import logging

from sgl_jax.srt.utils.jax_utils import get_device_name

logger = logging.getLogger(__name__)

DEFAULT_KDA_CHUNKING: tuple[int, int | None] = (64, None)

TUNED_KDA_CHUNKING: dict[str, dict[tuple[int, int, int], tuple[int, int | None]]] = {
    "TPU v7": {
        # GLM-5.3-Flash, TP=16: 64 linear-attn heads / 16 = 4 per shard,
        # head_dim = value_dim = 128.
        #
        # Tuned 2026-09-12 on zzl-tpu7x-slice-mig-0905 (v7x, 1 chip / 2 cores),
        # median of 10 iterations, one packed sequence, all tilings checked
        # against naive_recurrent_kda by
        # test_chunk_kda_intra_tilings_match_naive_recurrent_kda:
        #
        #     T        64/none    64/16    128/16    256/16
        #     2048       1.355    0.998     0.790     0.778
        #     8192       4.670    3.261     2.364     2.100
        #     16384      9.080    6.291     4.438     3.720
        #     65536     33.299   22.166    14.801    12.114
        #
        # 256/16 wins at every chunked-prefill bucket, by 2.4x at 16K. Without
        # sub-chunking neither 128 nor 256 compiles -- both hit the VMEM wall on
        # the [BT, BT, K] decay tensor -- so BT and BS have to be tuned as a
        # pair. BS=32 is consistently ~15% behind BS=16 once the inverse is off
        # the VPU (it was a tie before that), so the table does not carry it.
        #
        # A larger BT also costs padding: _align_seqs rounds every request up to
        # a multiple of BT, i.e. n_req*(BT-1) extra tokens per step. That is a
        # real tax but not a crossover -- at T=16384, BS=16, the ordering holds
        # at every request count (ms, 64/128/256):
        #
        #     n_req=1     6.274 / 4.442 / 3.709
        #     n_req=4     7.826 / 5.381 / 4.621
        #     n_req=16   14.487 / 9.573 / 7.727
        (4, 128, 128): (256, 16),
    },
}


def get_tuned_kda_chunking(num_heads: int, head_dim: int, value_dim: int) -> tuple[int, int | None]:
    """Best known ``(chunk_size, sub_chunk_size)`` for this per-shard shape.

    Falls back to ``(64, None)`` -- the pre-tuning behaviour -- on any miss.
    """
    table = TUNED_KDA_CHUNKING.get(get_device_name())
    if table is None:
        return DEFAULT_KDA_CHUNKING
    return table.get((num_heads, head_dim, value_dim), DEFAULT_KDA_CHUNKING)

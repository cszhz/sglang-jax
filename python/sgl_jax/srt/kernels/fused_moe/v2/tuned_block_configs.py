"""Auto-tuned block configs for fused_moe v2.

Same lookup approach as v1/tuned_block_configs.py but with v2's simpler
5-field FusedMoEBlockConfig (bt, bf, btc, bse, bts).
"""

# ruff: noqa: E501

from __future__ import annotations

import logging
import os

import jax.numpy as jnp

from sgl_jax.srt.utils.jax_utils import get_device_name

from .kernel import FusedMoEBlockConfig

logger = logging.getLogger(__name__)

# Key (without device_name):
#   (tokens_dtype, weight_dtype, num_tokens, num_experts, top_k,
#    hidden_size, intermediate_size, ep_size, use_shared_expert, use_grouped_topk,
#    enable_act_quant, quant_mode)
# enable_act_quant and quant_mode are optional trailing fields. quant_mode
# ("per_channel" | "blockwise" | "none") lets a shape store distinct winners for
# per-channel vs block-wise FP8, whose optimal tiles differ. Lookup is 3-level:
# the full key (with quant_mode) -> the key without quant_mode -> the legacy
# 10-field key (no enable_act_quant), so pre-existing entries still hit.
#
# Value: (bt, bf, btc, bse, bts)
# fmt: off
TUNED_BLOCK_CONFIGS: dict[str, dict[tuple, tuple[int, ...]]] = {
    "TPU v7": {
        # MiMo V2 Pro: E=384, H=6144, I=2048, top_k=8, fp8 e4m3, ep=32
        # Decode configs (tuned 2026-05-21)
        ('bfloat16', 'float8_e4m3fn', 64, 384, 8, 6144, 2048, 32, False, False): (8, 512, 8, 256, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 384, 8, 6144, 2048, 32, False, False): (8, 512, 8, 256, 8),
        ('bfloat16', 'float8_e4m3fn', 256, 384, 8, 6144, 2048, 32, False, False): (8, 512, 16, 256, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 384, 8, 6144, 2048, 32, False, False): (16, 1024, 32, 256, 32),
        ('bfloat16', 'float8_e4m3fn', 768, 384, 8, 6144, 2048, 32, False, False): (24, 1024, 32, 256, 32),
        # Prefill configs
        ('bfloat16', 'float8_e4m3fn', 2048, 384, 8, 6144, 2048, 32, False, False): (128, 512, 128, 256, None),
        ('bfloat16', 'float8_e4m3fn', 4096, 384, 8, 6144, 2048, 32, False, False): (128, 512, 128, 256, None),
        ('bfloat16', 'float8_e4m3fn', 8192, 384, 8, 6144, 2048, 32, False, False): (128, 1024, 56, 256, 112),
        ('bfloat16', 'float8_e4m3fn', 16384, 384, 8, 6144, 2048, 32, False, False): (256, 1024, 72, 256, 216),
        # MiMo V2 Pro with activation quantization, ep=128
        ('bfloat16', 'float8_e4m3fn', 512, 384, 8, 6144, 2048, 128, False, False, True): (8, 1024, 32, 256, 32),
        ('bfloat16', 'float8_e4m3fn', 768, 384, 8, 6144, 2048, 128, False, False, True): (8, 1024, 48, 256, 48),
        ('bfloat16', 'float8_e4m3fn', 2048, 384, 8, 6144, 2048, 128, False, False, True): (16, 1024, 64, 256, 64),
        ('bfloat16', 'float8_e4m3fn', 3072, 384, 8, 6144, 2048, 128, False, False, True): (24, 1024, 96, 256, 96),
        ('bfloat16', 'float8_e4m3fn', 4096, 384, 8, 6144, 2048, 128, False, False, True): (32, 1024, 64, 256, 128),
        ('bfloat16', 'float8_e4m3fn', 65536, 384, 8, 6144, 2048, 128, False, False, True): (256, 256, 8, 256, 856),
        # MiMo V2 Pro: E=384, H=6144, I=2048, top_k=8, fp8 e4m3, ep=8
        # Decode configs (tuned on bench-4 single-host v7x-16, 2026-05-21)
        ('bfloat16', 'float8_e4m3fn', 512, 384, 8, 6144, 2048, 8, False, False): (64, 1024, 32, 256, 32),
        # GLM-5.2: E=256, H=6144, I=2048, top_k=8, routed FP8 block-wise
        # K=128, ep=16, in-kernel shared expert, act_quant ON, no grouped top-k.
        # Tuned on 8 TPU v7x chips / 16 JAX devices, 2026-08-04
        # (Falcon exp-fvmfgcw2y9).
        ('bfloat16', 'float8_e4m3fn', 32, 256, 8, 6144, 2048, 16, True, False, True): (8, 512, 8, 128, 8),
        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 6144, 2048, 16, True, False, True): (8, 512, 8, 128, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 6144, 2048, 16, True, False, True): (8, 1024, 8, 128, 8),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 6144, 2048, 16, True, False, True): (16, 512, 16, 128, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 6144, 2048, 16, True, False, True): (32, 1024, 16, 1024, 32),
        ('bfloat16', 'float8_e4m3fn', 1024, 256, 8, 6144, 2048, 16, True, False, True): (64, 1024, 64, 1024, 64),
        ('bfloat16', 'float8_e4m3fn', 2048, 256, 8, 6144, 2048, 16, True, False, True): (128, 1024, 64, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 4096, 256, 8, 6144, 2048, 16, True, False, True): (128, 1024, 64, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 6144, 2048, 16, True, False, True): (128, 1024, 64, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 6144, 2048, 16, True, False, True): (256, 512, 64, 512, 256),
        # GLM-5.2: E=256, H=6144, I=2048, top_k=8, routed FP8 block-wise
        # K=128, ep=32, in-kernel shared expert, act_quant ON, no grouped top-k.
        # Tuned on 16 TPU v7x chips, 2026-08-03 (Falcon exp-bkbi8g86uy).
        ('bfloat16', 'float8_e4m3fn', 32, 256, 8, 6144, 2048, 32, True, False, True): (8, 512, 8, 128, 8),
        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 6144, 2048, 32, True, False, True): (8, 512, 8, 128, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 6144, 2048, 32, True, False, True): (8, 512, 16, 128, 16),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 6144, 2048, 32, True, False, True): (8, 512, 16, 128, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 6144, 2048, 32, True, False, True): (16, 1024, 32, 1024, 32),
        ('bfloat16', 'float8_e4m3fn', 1024, 256, 8, 6144, 2048, 32, True, False, True): (32, 1024, 64, 1024, 64),
        ('bfloat16', 'float8_e4m3fn', 2048, 256, 8, 6144, 2048, 32, True, False, True): (64, 1024, 40, 1024, 80),
        ('bfloat16', 'float8_e4m3fn', 4096, 256, 8, 6144, 2048, 32, True, False, True): (128, 1024, 32, 1024, 160),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 6144, 2048, 32, True, False, True): (128, 1024, 32, 1024, 160),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 6144, 2048, 32, True, False, True): (128, 1024, 32, 1024, 160),
        # GLM-5.3-Flash: E=288, H=4096, I=2048, top_k=8, routed FP8 block-wise
        # K=128, ep=16, in-kernel shared expert (I_se=2048), act_quant ON,
        # n_group=1 -> no grouped top-k.
        # Tuned 2026-09-10 on 2 hosts of mig-0905 (8 v7x chips / 16 JAX devices)
        # with bench_v2.py BENCH_TUNE=1 BENCH_MAX_CONFIGS=40. Measured twice,
        # agreeing to 0.003 ms. 11-field key (no quant_mode) so blockwise and
        # per-channel both land here; this shape only ships blockwise.
        #
        # Why it matters: without these the lookup silently falls through to
        # DEFAULT_V2_BLOCK_CONFIG (32/512/32/256), which is tuned for H=6144
        # and badly under-blocks H=4096. 1024 tok 0.491 -> 0.296 ms (1.66x),
        # 16384 tok 7.858 -> 4.575 ms (1.72x). At 200K ctx / dp=1 the 16384
        # bucket is the whole prefill path, where MoE was 24% of the step.
        # Decode buckets (tuned 2026-09-12, same harness, 15 candidates each).
        # These were the last silent DEFAULT_V2_BLOCK_CONFIG fallthroughs on
        # this shape.
        #
        # Honest sizing: the win here is small, 1.01-1.05x, not the 1.7x the
        # prefill buckets got. At these token counts the tuner clamps bt/btc to
        # 8 regardless, so the only free parameters left are bf and bse, and
        # DEFAULT's bf=512/bse=256 is already nearly right:
        #
        #     tokens   default-shape   best
        #        16      0.124 ms      0.119 ms  bf=1024 bse=512
        #        32      0.178 ms      0.170 ms  bf=1024 bse=512
        #        64      0.187 ms      0.179 ms  bf=1024 bse=512
        #       128      0.209 ms      0.207 ms  bf=512  bse=512
        #
        # MoE is ~12% of decode device time, so this is ~0.5% of TPOT. Worth
        # having because it is measured and costs nothing, not because it moves
        # the number.
        ('bfloat16', 'float8_e4m3fn', 16, 288, 8, 4096, 2048, 16, True, False, True): (8, 1024, 8, 512, 8),
        ('bfloat16', 'float8_e4m3fn', 32, 288, 8, 4096, 2048, 16, True, False, True): (8, 1024, 8, 512, 8),
        ('bfloat16', 'float8_e4m3fn', 64, 288, 8, 4096, 2048, 16, True, False, True): (8, 1024, 8, 512, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 288, 8, 4096, 2048, 16, True, False, True): (8, 512, 8, 512, 8),
        ('bfloat16', 'float8_e4m3fn', 1024, 288, 8, 4096, 2048, 16, True, False, True): (64, 2048, 64, 1024, 64),
        # 16384: bt=256/btc=128 (4.293) edges out the symmetric bt=btc=128
        # (4.578). Six of the 40 candidates (bt>=512 with bf>=1024) VMEM-OOM
        # during autotune and are simply skipped -- that is the tuner working,
        # not a failure.
        ('bfloat16', 'float8_e4m3fn', 16384, 288, 8, 4096, 2048, 16, True, False, True): (256, 1024, 128, 1024, 256),
        # Same shape, shared expert run OUTSIDE the kernel (see
        # layers/fused_moe._shared_expert_dense). Re-swept 2026-09-13 rather
        # than carried over: the in-kernel entry above was the best of a
        # *smaller* feasible set, so it says nothing about this one. 48 of 594
        # valid candidates, 28 measured / 7 VMEM-or-SMEM OOM.
        #   3.797  in-kernel (the entry above, shared expert included)
        #   3.685  routed-only at that same block config
        #   2.671  bse 1024 -> 256: the freed VMEM is the larger half of the win
        #   2.486  + bt 256 -> 512, btc 128 -> 256   <- this entry
        # Runner-up (256, 1024, 144, 256, 144) at 2.503 is within noise of the
        # winner, so bt=512 is not load-bearing on its own -- bse is.
        ('bfloat16', 'float8_e4m3fn', 16384, 288, 8, 4096, 2048, 16, False, False, True): (512, 1024, 256, 256, 256),
        # 65536 = chunked_prefill 16384 x dp_size 4, i.e. the dp=4 prefill bucket.
        # Same winner shape as 16384. 31.407 -> 17.066 ms (1.84x).
        # 32768 (the dp=2 bucket) is not swept yet and still falls back to DEFAULT.
        ('bfloat16', 'float8_e4m3fn', 65536, 288, 8, 4096, 2048, 16, True, False, True): (256, 1024, 128, 1024, 256),
        # 131072 = the dp=8 prefill bucket. Same winner again (33.973 ms; the
        # next candidate that satisfies bse<=bf and bt<=bts is 55.966). Note
        # 65536 -> 131072 is 17.066 -> 33.973 ms, i.e. exactly linear: MoE has
        # no economy of scale here, so every padded token is pure loss.
        ('bfloat16', 'float8_e4m3fn', 131072, 288, 8, 4096, 2048, 16, True, False, True): (256, 1024, 128, 1024, 256),
        # Ling 2.6-1T: E=256, H=8192, I=2048, top_k=8, fp8 e4m3 per-channel, ep=32
        # Tuned 2026-05-27
        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 8192, 2048, 32, False, True): (8, 256, 8, 256, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 8192, 2048, 32, False, True): (8, 256, 8, 256, 8),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 8192, 2048, 32, False, True): (8, 256, 16, 256, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 8192, 2048, 32, False, True): (16, 1024, 32, 256, 32),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 8192, 2048, 32, False, True): (128, 512, 160, 256, 160),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 8192, 2048, 32, False, True): (128, 512, 80, 256, 160),
        # Ling 2.6-1T with in-kernel shared expert, act_quant ON (Mode 1: fp8 token
        # x fp8 weight). 11-field key (last = enable_act_quant); use_grouped_topk=True
        # because Ling routes with n_group=8. Tuned 2026-05-31. SE-on vs no-SE: same
        # routed shape, bse maxed (per-block SE weight-DMA is the dominant SE cost).
        # act-OFF (W8A16, bf16 token) is a different VMEM regime, not tuned here ->
        # falls back to the legacy key / DEFAULT.
        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 8192, 2048, 32, True, True, True): (8, 256, 8, 128, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 8192, 2048, 32, True, True, True): (8, 512, 16, 128, 16),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 8192, 2048, 32, True, True, True): (8, 256, 16, 128, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 8192, 2048, 32, True, True, True): (16, 1024, 32, 1024, 32),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 8192, 2048, 32, True, True, True): (128, 512, 80, 512, 160),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 8192, 2048, 32, True, True, True): (128, 512, 80, 512, 160),
    },
    "*": {},
}
# fmt: on

DEFAULT_V2_BLOCK_CONFIG = FusedMoEBlockConfig(
    bt=32,
    bf=512,
    btc=32,
    bse=256,
)


def should_interleave_fused_moe_v2_bt(*, num_tokens: int, ep_size: int) -> bool:
    """Deprecated shim: gather overlap is now unconditional for num_bt > 1 via
    the kernel's fixed-K gather banks, so there is no token-count gate. Retained
    only to validate inputs for callers that still probe it; always returns True.
    """
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")
    if num_tokens % ep_size != 0:
        raise ValueError(f"Expected {num_tokens=} to be aligned to {ep_size=}.")
    return True


def get_simplified_key(
    *,
    dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    ep_size: int,
    use_shared_expert: bool,
    use_grouped_topk: bool,
    enable_act_quant: bool = False,
    quant_mode: str | None = None,
) -> tuple:
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")
    if num_tokens % ep_size != 0:
        raise ValueError(f"Expected {num_tokens=} to be aligned to {ep_size=}.")
    if quant_mode not in (None, "none", "blockwise", "per_channel"):
        raise ValueError(f"Unsupported {quant_mode=}")

    device = get_device_name()
    dtype_name = jnp.dtype(dtype).name
    weight_dtype_name = jnp.dtype(weight_dtype).name
    return (
        device,
        dtype_name,
        weight_dtype_name,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        ep_size,
        bool(use_shared_expert),
        bool(use_grouped_topk),
        bool(enable_act_quant),
        quant_mode,
    )


def _env_override(num_tokens: int) -> tuple[int, ...] | None:
    """Override one bucket's block config from the environment, for in-situ sweeps.

    The table above was swept with bench_v2.py, which feeds the kernel a
    *balanced* token-to-expert distribution. Real routing is not balanced, and
    the kernel's cost is sum(ceil(count_e / bt)) blocks -- so a large ``bt`` is
    free in the bench and expensive in production. Re-sweeping inside the real
    server is the only way to see that, and editing the table for each candidate
    means a source change per data point.

    Format: ``SGLANG_JAX_MOE_V2_BLOCK_CFG=<num_tokens>:bt,bf,btc,bse,bts``,
    semicolon-separated for several buckets. ``bts`` may be ``none``.
    """
    spec = os.environ.get("SGLANG_JAX_MOE_V2_BLOCK_CFG", "").strip()
    if not spec:
        return None
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        bucket, _, values = part.partition(":")
        if int(bucket) != num_tokens:
            continue
        fields = [None if v.strip().lower() == "none" else int(v) for v in values.split(",")]
        if len(fields) != 5:
            raise ValueError(f"SGLANG_JAX_MOE_V2_BLOCK_CFG: {part!r} needs 5 fields")
        logger.warning("v2 block config for num_tokens=%d overridden to %s", num_tokens, fields)
        return tuple(fields)
    return None


def get_tuned_fused_moe_v2_block_config(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    ep_size: int,
    use_shared_expert: bool = False,
    use_grouped_topk: bool = False,
    enable_act_quant: bool = False,
    quant_mode: str | None = None,
) -> FusedMoEBlockConfig:
    keys = get_simplified_key(
        dtype=dtype,
        weight_dtype=weight_dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ep_size=ep_size,
        use_shared_expert=use_shared_expert,
        use_grouped_topk=use_grouped_topk,
        enable_act_quant=enable_act_quant,
        quant_mode=quant_mode,
    )
    device_name = keys[0]
    # 3-level fallback: full key (with quant_mode) -> without quant_mode ->
    # legacy 10-field key (no enable_act_quant), so pre-existing entries still hit.
    table_key_quant = keys[1:]
    table_key = table_key_quant[:-1]
    table_key_legacy = table_key[:-1]

    def _lookup(k):
        cfg = None
        if device_name in TUNED_BLOCK_CONFIGS:
            cfg = TUNED_BLOCK_CONFIGS[device_name].get(k)
        if cfg is None:
            cfg = TUNED_BLOCK_CONFIGS.get("*", {}).get(k)
        return cfg

    cfg_tuple = _lookup(table_key_quant)
    if cfg_tuple is None:
        cfg_tuple = _lookup(table_key)
    if cfg_tuple is None:
        cfg_tuple = _lookup(table_key_legacy)

    override = _env_override(num_tokens)
    if override is not None:
        cfg_tuple = override

    if cfg_tuple is None:
        return DEFAULT_V2_BLOCK_CONFIG

    if len(cfg_tuple) != 5:
        raise ValueError(f"Unexpected v2 tuned config tuple length: {len(cfg_tuple)}")

    bt, bf, btc, bse, bts = cfg_tuple
    logger.info(
        "Using v2 tuned block config: num_tokens=%d bt=%d bf=%d btc=%d bse=%d bts=%s",
        num_tokens,
        bt,
        bf,
        btc,
        bse,
        bts,
    )

    return FusedMoEBlockConfig(bt=bt, bf=bf, btc=btc, bse=bse, bts=bts)

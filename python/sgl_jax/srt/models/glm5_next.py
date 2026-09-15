"""GLM-5.3-Flash (``glm5_next``) text model.

Three things make this model different from GLM-5.2 (``glm5_moe.py``), and all
three are load-bearing:

1. **mHC** (manifold-constrained hyper-connections). The residual stream is
   ``hc_mult=4`` wide for the whole depth of the model. Every layer opens with a
   learned collapse of the 4 streams into one vector and closes with a learned
   scatter back into 4, mixed by a Sinkhorn-normalized ``4x4`` matrix. See
   ``Glm5NextHyperConnection``.
2. **NoPE MLA**. ``qk_rope_head_dim == 0``: there is no positional encoding in
   the full-attention layers at all. See ``Glm5NextAttention``.
3. **Hybrid attention**. 34 of 45 layers are KDA (Kimi Delta Attention) with a
   *bounded* gate; the other 11 are MLA + DSA indexer. The KDA half is
   structurally identical to ``kimi_linear.py``'s.

Not yet implemented:
  - the DSA indexer's k-pooling. Its parameters are declared so the checkpoint
    loads, but ``use_dsa_sparse`` must stay off; the full-attention layers fall
    back to dense MLA over the whole context.
  - the vision tower (``model.visual.*``, 347 tensors). Those checkpoint keys
    are simply never read.
"""

from __future__ import annotations

import functools
import logging
import os

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.glm5_next import Glm5NextConfig, Glm5NextTextConfig
from sgl_jax.srt.configs.model_config import ModelConfig, MoEBackend
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.kernels.hyper_connection import hc_weights_pallas
from sgl_jax.srt.layers.attention.fla.gated_rmsnorm import GatedRMSNorm
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import (
    FusedEPMoEV2,
    GateLogit,
    TopK,
    create_moe_weights_mapping,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.parallel_utils import make_reduce_sharding
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

# GLM-5.3 is NoPE. The absorbed-MLA path used to be built around a rope tail,
# so we fed it an all-zero pe of width 64 — exactly equivalent arithmetically
# (q_pe·k_pe = 0 for every pair, and the softmax scale comes from
# ``layer.scaling``, not the head dim), but it cost a *padded* 128 lanes per
# token per MLA layer in the KV cache: the pool allocates
# ``align_to(512,128) + align_to(64,128) = 640``, i.e. 20% of KV spent on
# provably-zero values, plus the DMAs and the QK lanes to match.
#
# Both the backend (``mla_backend.py``) and the v2 Pallas kernel now accept
# ``q_rope=k_rope=None`` and skip every pe DMA/concat, so the width is 0: the
# pool allocates 512 and the kernel does a pure-latent QK.
#
# ``patch_model_config`` writes this same value into
# ``hf_text_config.qk_rope_head_dim`` so ``MLATokenToKVPool`` agrees with the
# kernel.
_NOPE_PE_WIDTH = 0


def _unweighted_rms_scale(x: jax.Array, eps: float) -> jax.Array:
    """Per-row ``rsqrt(mean(x^2) + eps)`` of ``Glm5NextTextUnweightedRMSNorm``.

    The norm carries no learned scale, so it is nothing but a per-token scalar
    rescale. Returning that scalar instead of the rescaled tensor lets the one
    caller apply it *after* the projection — see
    ``Glm5NextHyperConnection.__call__`` for why that matters.

    The reduction accumulates in fp32 like the reference. The fp32 cast feeds
    only the reduce, so it never lands in HBM.
    """
    mean_sq = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return jax.lax.rsqrt(mean_sq + eps)


class Glm5NextHyperConnection(nnx.Module):
    """One mHC site (``attn_hc`` or ``ffn_hc``) of one layer.

    Reads the ``[T, hc, D]`` residual streams and produces

      * ``pre``      ``[T, hc]``      — weights collapsing the streams into the
                                        block input,
      * ``post``     ``[T, hc]``      — weights scattering the block output back,
      * ``comb``     ``[T, hc, hc]``  — doubly-stochastic (Sinkhorn) stream mixer.

    All three come from a single ``[hc*D] -> (2+hc)*hc`` projection of the
    normalized flattened streams, so the checkpoint stores one ``fn`` matrix per
    site plus a per-output ``base`` bias and a 3-entry ``scale``.

    The weight math runs in fp32 (the reference does the same) — the Sinkhorn
    iteration divides by row/column sums 20 times and bf16 drifts. That applies
    from ``mixed`` onward, where the tensors are ``[T, 24]`` and ``[T, hc, hc]``
    and fp32 is free; that whole tail lives in ``hc_weights_pallas``. It
    deliberately does *not* apply to the full-width ``[T, hc*D]`` input:
    upcasting that is pure HBM traffic, since it is bf16 data either way. See
    ``__call__``.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int,
        hc_sinkhorn_iters: int,
        hc_eps: float,
        rms_norm_eps: float,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.rms_norm_eps = rms_norm_eps
        self.mesh = mesh

        self.mix_size = (2 + hc_mult) * hc_mult
        self.flat_size = hc_mult * hidden_size

        # Stored transposed relative to the checkpoint ([mix, hc*D] on disk) so
        # the forward is a plain ``flat @ fn``. Replicated: 16384x24 bf16 is
        # 786 KB, and the input ``flat`` is replicated across the tensor axis
        # anyway. Plain nnx.Param (not LinearBase) keeps quantize_model away
        # from it — the checkpoint ships these in bf16/fp32 and lists them in
        # modules_to_not_convert.
        self.fn = nnx.Param(
            jnp.zeros((self.flat_size, self.mix_size), dtype=dtype, out_sharding=P(None, None))
        )
        self.base = nnx.Param(jnp.zeros((self.mix_size,), dtype=jnp.float32, out_sharding=P(None)))
        self.scale = nnx.Param(jnp.zeros((3,), dtype=jnp.float32, out_sharding=P(None)))

    def __call__(self, streams: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """``streams: [T, hc, D]`` -> ``(post [T, hc], comb [T, hc, hc], collapsed [T, D])``."""
        hc = self.hc_mult
        eps = self.hc_eps
        num_tokens = streams.shape[0]

        flat = streams.reshape(num_tokens, self.flat_size)
        # Fold the RMS scale past the projection: the norm is a per-token scalar
        # and the projection is linear, so ``(x*s) @ fn == (x @ fn) * s``.
        # Normalizing first would materialize ``flat`` in fp32 — 1.07 GB per site
        # at T=16384, and a forward pass runs 90 sites — for no precision at all:
        # ``streams`` and ``fn`` are both bf16, so the upcast hands the MXU no
        # bits it did not already have, and bf16 x bf16 accumulates into fp32
        # natively. Everything downstream of ``mixed`` (in particular the 20
        # Sinkhorn divisions) stays in fp32 exactly as before.
        inv_rms = _unweighted_rms_scale(flat, self.rms_norm_eps)
        mixed = (
            jnp.einsum("tf,fm->tm", flat, self.fn.value, preferred_element_type=jnp.float32)
            * inv_rms
        )

        # sigmoid pair + softmax + 39 Sinkhorn reduce/divide pairs, all on
        # [T, 4, 4] tensors, fused into one op. Written out in XLA this tail is
        # ~80 kernel launches per site and 90 sites run per forward; at decode
        # the tensors are a few hundred bytes each, so that is ~7000 launches
        # of pure overhead and the single largest term in TPOT. See
        # ``hyper_connection/v1/kernel.py``.
        fn = functools.partial(
            hc_weights_pallas,
            hc=hc,
            eps=eps,
            iters=self.hc_sinkhorn_iters,
            out_dtype=streams.dtype,
        )
        if self.mesh is not None:
            # Mosaic kernels are opaque to the SPMD partitioner, so the call has
            # to see manual mesh axes. The weights are replicated; the token
            # axis follows whatever ``mixed`` already carries — P("data") under
            # dp>1, unsharded under dp=1 — and under explicit sharding the
            # in_specs have to match that exactly rather than assume.
            tok = jax.typeof(mixed).sharding.spec[0]
            fn = jax.shard_map(
                fn,
                mesh=self.mesh,
                in_specs=(P(tok, None), P(None), P(None)),
                out_specs=(P(tok, None), P(tok, None), P(tok, None, None)),
                check_vma=False,
            )
        pre, post, comb = fn(mixed, self.scale.value, self.base.value)

        # The fp32 cast looks like it should materialize 1.07 GB per site at
        # T=16384, the way normalizing ``flat`` early would above -- but it does
        # not: XLA folds the convert into the dot, and dropping it in favour of
        # a bf16 dot with ``preferred_element_type=jnp.float32`` measures the
        # same to 0.3% and is bit-identical (benchmark/kernels/hyper_connection/
        # bench_hc_combine.py). Left as-is.
        collapsed = jnp.einsum("th,thd->td", pre, streams.astype(jnp.float32))
        return post, comb, collapsed.astype(streams.dtype)


def _hc_combine(
    post: jax.Array,
    comb: jax.Array,
    block_out: jax.Array,
    residual_streams: jax.Array,
) -> jax.Array:
    """Scatter a block output back into the 4-wide residual stream.

    Reference (torch, ``[B, T, hc, D]``)::

        post.unsqueeze(-1) * block_out.unsqueeze(-2)
            + torch.matmul(comb.transpose(-1, -2), residual)

    The transposed matmul is ``out[k, d] = sum_h comb[h, k] * residual[h, d]``.
    """
    mixed = jnp.einsum("thk,thd->tkd", comb, residual_streams)
    return post[:, :, None] * block_out[:, None, :] + mixed


class Glm5NextMLP(nnx.Module):
    """Dense FFN for the first ``first_k_dense_replace`` layers.

    GLM-5.3 clamps the SwiGLU *before* the activation::

        silu(min(gate, limit)) * clip(up, -limit, limit)

    which is not what the fused MoE kernel does (it clamps ``silu(gate)``
    after the fact). The difference is ~5e-4 at ``swiglu_limit=10``, but the
    dense path is cheap to get exactly right, so it is written out here rather
    than routed through ``apply_fused_mlp_with_padding``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        swiglu_limit: float | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.swiglu_limit = swiglu_limit
        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )
        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )

    def __call__(
        self, hidden_states: jax.Array, out_sharding: jax.sharding.Sharding | None = None
    ) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        if self.swiglu_limit is not None:
            limit = jnp.asarray(self.swiglu_limit, dtype=gate.dtype)
            gate = jnp.minimum(gate, limit)
            up = jnp.clip(up, -limit, limit)
        output, _ = self.down_proj(jax.nn.silu(gate) * up, out_sharding=out_sharding)
        return output


class Glm5NextKdaAttention(nnx.Module):
    """KDA layer.

    Parameter-for-parameter the same as ``kimi_linear.KimiDeltaAttention`` —
    the GLM-5.3 checkpoint ships the sglang/vLLM-converted layout (split
    ``q/k/v_conv1d``, flat ``A_log``/``dt_bias``), not the fused ``conv1d`` +
    nested ``forget_gate`` of the HF reference file.

    Two reasons this is its own class instead of a reuse:

      * ``KimiDeltaAttention`` derives ``v_head_dim`` from
        ``getattr(config, "v_head_dim")``, which on GLM-5.3 is the *MLA* value
        (256), not KDA's 128. KDA here is symmetric: q, k and v are all
        ``64 x 128``.
      * the gate is bounded — ``-5.0 * sigmoid(exp(A_log) * (g + dt_bias))``
        instead of Kimi's ``-exp(A_log) * softplus(g + dt_bias)``. That form
        already exists in the Pallas kernel; ``kda_lower_bound`` on
        ``RadixLinearAttention`` selects it for both prefill and decode.
        The HF config spells the same knob ``gate_lower_bound``.
    """

    def __init__(
        self,
        config: Glm5NextTextConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        linear_config = config.linear_attn_config
        self.mesh = mesh
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.conv_size = linear_config["short_conv_kernel_size"]
        self.head_dim = linear_config["head_dim"]
        self.num_heads = linear_config["num_heads"]
        self.projection_size = self.num_heads * self.head_dim
        self.gate_lower_bound = linear_config.get("gate_lower_bound")

        def _linear(in_size, out_size, kernel_axes, scope_name):
            return LinearBase(
                input_size=in_size,
                output_size=out_size,
                kernel_axes=kernel_axes,
                use_bias=False,
                params_dtype=dtype,
                mesh=mesh,
                scope_name=scope_name,
            )

        self.q_proj = _linear(self.hidden_size, self.projection_size, (None, "tensor"), "q_proj")
        self.k_proj = _linear(self.hidden_size, self.projection_size, (None, "tensor"), "k_proj")
        self.v_proj = _linear(self.hidden_size, self.projection_size, (None, "tensor"), "v_proj")

        # Depthwise short-conv weights, kept as ``[D, K]`` — the layout
        # ``short_convolution`` consumes. LinearBase is only a parameter
        # container here and is never called.
        self.q_conv1d = _linear(self.projection_size, self.conv_size, ("tensor", None), "q_conv1d")
        self.k_conv1d = _linear(self.projection_size, self.conv_size, ("tensor", None), "k_conv1d")
        self.v_conv1d = _linear(self.projection_size, self.conv_size, ("tensor", None), "v_conv1d")

        self.A_log = nnx.Param(
            jnp.zeros(
                (1, 1, self.num_heads, 1),
                dtype=jnp.float32,
                out_sharding=P(None, None, "tensor", None),
            )
        )
        self.dt_bias = nnx.Param(
            jnp.zeros((self.projection_size,), dtype=jnp.float32, out_sharding=P("tensor"))
        )

        self.f_a_proj = _linear(self.hidden_size, self.head_dim, (None, None), "f_a_proj")
        self.f_b_proj = _linear(self.head_dim, self.projection_size, (None, "tensor"), "f_b_proj")
        self.g_a_proj = _linear(self.hidden_size, self.head_dim, (None, None), "g_a_proj")
        self.g_b_proj = _linear(self.head_dim, self.projection_size, (None, "tensor"), "g_b_proj")
        self.b_proj = _linear(self.hidden_size, self.num_heads, (None, "tensor"), "b_proj")

        self.o_norm = GatedRMSNorm(self.head_dim, epsilon=config.rms_norm_eps)
        self.o_proj = _linear(self.projection_size, self.hidden_size, ("tensor", None), "o_proj")

        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_heads,
            num_k_heads=self.num_heads,
            num_v_heads=self.num_heads,
            head_q_dim=self.head_dim,
            head_k_dim=self.head_dim,
            head_v_dim=self.head_dim,
            q_conv1d=self.q_conv1d,
            k_conv1d=self.k_conv1d,
            v_conv1d=self.v_conv1d,
            bias=None,
            activation=jax.nn.silu,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            kda_lower_bound=self.gate_lower_bound,
        )

    @named_scope("kda")
    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[jax.Array, object]:
        num_tokens = hidden_states.shape[0]

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        raw_gate, _ = self.f_b_proj(self.f_a_proj(hidden_states)[0])
        raw_gate = raw_gate.reshape(num_tokens, self.num_heads, self.head_dim)
        beta = jax.nn.sigmoid(self.b_proj(hidden_states)[0].astype(jnp.float32))

        o, recurrent_state_pool = self.attn(
            forward_batch, q, k, v, raw_gate, beta, recurrent_state_pool
        )
        o = o.reshape(num_tokens, self.num_heads, self.head_dim)

        output_gate, _ = self.g_b_proj(self.g_a_proj(hidden_states)[0])
        output_gate = output_gate.reshape(num_tokens, self.num_heads, self.head_dim)
        o = self.o_norm(o, output_gate).reshape(num_tokens, self.projection_size)
        o, _ = self.o_proj(o, out_sharding=out_sharding)
        return o, recurrent_state_pool


class Glm5NextIndexer(nnx.Module):
    """DSA indexer — **parameters only, no forward yet**.

    Declared so the checkpoint loads end-to-end while the k-pooling top-k is
    still unimplemented. Until then the DSA layers run
    dense MLA over the full context: slower, but numerically the ground truth
    the sparse path has to reproduce.

    GLM-5.3's indexer differs from GLM-5.2's in three ways, all of which the
    eventual kernel work has to honour:

      * no RoPE and no Hadamard rotation on the index query/key;
      * ``k_norm`` is a real LayerNorm (weight *and* bias, eps 1e-6);
      * keys are pooled in groups of ``index_kpool=4`` before scoring, with
        ``index_kpool_compress_{ape,gate}`` producing the pooling weights, so
        ``index_topk=2048`` means 512 pools plus a raw tail.
    """

    def __init__(
        self,
        config: Glm5NextTextConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.mesh = mesh
        self.head_dim = config.index_head_dim
        self.n_head = config.index_n_heads
        self.index_topk = config.index_topk
        self.index_kpool = config.index_kpool

        self.wq_b = LinearBase(
            input_size=config.q_lora_rank,
            output_size=config.index_head_dim * config.index_n_heads,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="wq_b",
        )
        self.wk = LinearBase(
            input_size=config.hidden_size,
            output_size=config.index_head_dim,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="wk",
        )
        self.weights_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=config.index_n_heads,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="weights_proj",
        )
        # LayerNorm, not RMSNorm: mean-centered, with a bias. eps is 1e-6 in
        # the reference, independent of config.rms_norm_eps (1e-5).
        self.k_norm_weight = nnx.Param(
            jnp.ones((config.index_head_dim,), dtype=dtype, out_sharding=P(None))
        )
        self.k_norm_bias = nnx.Param(
            jnp.zeros((config.index_head_dim,), dtype=dtype, out_sharding=P(None))
        )
        self.k_norm_eps = 1e-6

        # k-pooling: a learned per-slot positional bias over the 4 pooled
        # positions, plus a gate projection scoring each key inside its pool.
        self.index_kpool_compress_ape = nnx.Param(
            jnp.zeros(
                (config.index_kpool, config.index_head_dim), dtype=dtype, out_sharding=P(None, None)
            )
        )
        self.index_kpool_compress_gate = LinearBase(
            input_size=config.hidden_size,
            output_size=config.index_head_dim,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="index_kpool_compress_gate",
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "GLM-5.3 indexer k-pooling is not implemented yet; run with dense MLA "
            "(use_dsa_sparse=False)."
        )


class Glm5NextAttention(nnx.Module):
    """NoPE MLA, absorbed.

    Same absorbed structure as ``glm5_moe.Glm5Attention`` minus everything
    positional: no rotary embedding, no q/k norm, and ``kv_a_proj_with_mqa``
    emits only the ``kv_lora_rank``-wide latent (the checkpoint's
    ``[512, 4096]`` confirms there is no rope tail to split off). The zero pe
    fed to the backend is explained at ``_NOPE_PE_WIDTH``.
    """

    def __init__(
        self,
        config: Glm5NextTextConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        use_dsa_sparse: bool = False,
    ):
        self.mesh = mesh
        self.layer_id = layer_id
        self.dtype = dtype
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = self.qk_head_dim**-0.5
        self.use_dsa_sparse = use_dsa_sparse

        # NoPE invariant: the whole query head is the nope part. Deliberately
        # *not* asserted on ``qk_rope_head_dim`` — patch_model_config has
        # already rewritten that field to _NOPE_PE_WIDTH for KV-pool sizing by
        # the time any layer is constructed.
        if not config.mla_use_nope or config.qk_head_dim != config.qk_nope_head_dim:
            raise ValueError(
                "Glm5NextAttention is the NoPE variant; got "
                f"mla_use_nope={config.mla_use_nope}, "
                f"qk_head_dim={config.qk_head_dim}, "
                f"qk_nope_head_dim={config.qk_nope_head_dim}"
            )

        self.q_a_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=self.q_lora_rank,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="q_a_layernorm",
        )
        self.q_b_proj = LinearBase(
            input_size=self.q_lora_rank,
            output_size=self.num_heads * self.qk_head_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_b_proj",
        )
        self.kv_a_proj_with_mqa = LinearBase(
            input_size=config.hidden_size,
            output_size=self.kv_lora_rank,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="kv_a_layernorm",
        )
        self.kv_b_proj = LinearBase(
            input_size=self.kv_lora_rank,
            output_size=self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="kv_b_proj",
        )
        self.o_proj = LinearBase(
            input_size=self.num_heads * self.v_head_dim,
            output_size=config.hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )

        self.indexer = Glm5NextIndexer(config, layer_id=layer_id, mesh=mesh, dtype=dtype)

        uk_axes = (None, "tensor", None)
        self.w_uk = nnx.Param(
            jnp.zeros(
                (self.kv_lora_rank, self.num_heads, self.qk_nope_head_dim),
                dtype=dtype,
                out_sharding=P(*uk_axes),
            )
        )
        self.w_uv = nnx.Param(
            jnp.zeros(
                (self.kv_lora_rank, self.num_heads, self.v_head_dim),
                dtype=dtype,
                out_sharding=P(*uk_axes),
            )
        )
        self.attn_mqa = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.kv_lora_rank + _NOPE_PE_WIDTH,
            scaling=self.scaling,
            num_kv_heads=1,
            v_head_dim=self.kv_lora_rank,
            layer_id=layer_id,
        )

    def post_load_weights(self):
        """Fold ``kv_b_proj`` into the absorbed ``w_uk``/``w_uv``.

        GLM-5.3 keeps ``kv_b_proj`` in BF16 (it is in ``modules_to_not_convert``
        for every DSA layer), so unlike GLM-5.2 there is no FP8 dequant branch
        to take here.
        """
        if self.kv_b_proj is None:
            return
        if not hasattr(self.kv_b_proj, "weight"):
            raise ValueError(
                f"layer {self.layer_id}: kv_b_proj is quantized, but GLM-5.3 ships it "
                "in BF16. Check quantization_config.modules_to_not_convert."
            )
        w_kv = self.kv_b_proj.weight.value.reshape(
            self.kv_lora_rank, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        self.w_uk.value = w_kv[:, :, : self.qk_nope_head_dim]
        self.w_uv.value = w_kv[:, :, self.qk_nope_head_dim :]
        self.kv_b_proj = None

    @named_scope("nope_mla")
    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if self.use_dsa_sparse:
            raise NotImplementedError(
                "GLM-5.3 DSA sparse attention needs the k-pooling indexer, which is not implemented yet."
            )
        num_tokens = hidden_states.shape[0]

        q_compressed, _ = self.q_a_proj(hidden_states)
        q_compressed = self.q_a_layernorm(q_compressed)
        q, _ = self.q_b_proj(q_compressed)
        q_nope = q.reshape(num_tokens, self.num_heads, self.qk_head_dim)

        compressed, _ = self.kv_a_proj_with_mqa(hidden_states)
        compressed = self.kv_a_layernorm(compressed)

        # "thd,rhd->thr" — fp32 accumulate, as in glm5_moe: the bf16
        # accumulator on this small batched dot drifts enough over 45 layers to
        # push decode into repetition.
        ql_nope = jax.lax.dot_general(
            q_nope,
            self.w_uk.value,
            (((2,), (2,)), ((1,), (1,))),
            preferred_element_type=jnp.float32,
        ).astype(q_nope.dtype)
        ql_nope = ql_nope.transpose(1, 0, 2)

        c_kv_3d = compressed[:, None, :]
        attn_output, kv_fused = self.attn_mqa(
            ql_nope,
            c_kv_3d,
            c_kv_3d,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            # NoPE: no q_rope/k_rope at all — see _NOPE_PE_WIDTH.
        )

        # "thr,rhd->thd" — fp32 accumulate; see ql_nope above.
        o_v = jax.lax.dot_general(
            attn_output,
            self.w_uv.value,
            (((2,), (0,)), ((1,), (1,))),
            preferred_element_type=jnp.float32,
        ).astype(attn_output.dtype)
        o_v = o_v.transpose(1, 0, 2).reshape(num_tokens, self.num_heads * self.v_head_dim)

        output, _ = self.o_proj(o_v, out_sharding=out_sharding)
        return output, kv_fused


# Benchmarking knob, not a serving option: overwrite the router's decision so the
# fused MoE kernel can be priced under a controlled expert assignment *in situ*.
# The model's output is garbage under any non-empty value. It exists because the
# standalone kernel bench (kernels/fused_moe/v2/bench_v2.py) does not reproduce
# production timings -- it predicted 1.85x from rebalancing where the real gain
# was 1.13x -- so the only trustworthy way to size the remaining headroom is to
# ablate the routing inside the real server.
#
# Three modes, because "balanced" and "local" are different properties and the
# first measurement could not tell them apart: the static permutation moved the
# imbalance metric 2.81 -> 1.43 for 1.13x, yet mode 1 (metric 1.00) gave 2.25x.
# Time cannot be that non-linear in one variable, so mode 1 must also be buying
# something other than balance. The three modes vary the two properties
# independently:
#
#   1  counts exactly equal, expert ids consecutive   (balanced + maximally local)
#   2  counts exactly equal, expert ids strided       (balanced, locality broken)
#   3  real counts preserved exactly, tokens shuffled (unbalanced, locality broken)
#
# 3 vs. production isolates locality; 1/2 vs. 3 isolates balance. Only the
# balance part is reachable by redundant experts or a better EPLB, so this is
# what decides whether --ep-num-redundant-experts is worth its cost.
_DEBUG_BALANCED_ROUTING = os.environ.get("SGLANG_JAX_DEBUG_BALANCED_ROUTING", "").strip()
if _DEBUG_BALANCED_ROUTING in ("", "0"):
    _DEBUG_BALANCED_ROUTING = ""


def _shard_local_stride_shuffle(ids: jax.Array) -> jax.Array:
    """Permute the flat (tokens, slots) axis of one device's shard by a stride."""
    n, k = ids.shape
    m = n * k
    idx = (jax.lax.iota(jnp.int32, m) * 7919) % m
    return jnp.take(ids.reshape(-1), idx, axis=0).reshape(n, k)


def _round_robin_topk_ids(
    topk_ids: jax.Array,
    num_physical_experts: int,
    mode: str,
    *,
    mesh=None,
    token_axes="data",
) -> jax.Array:
    """Replace routing with a synthetic assignment; see ``_DEBUG_BALANCED_ROUTING``."""
    n, k = topk_ids.shape
    if mode == "3":
        # Reorder the real ids by a stride permutation. A permutation keeps the
        # per-expert counts bit-exact -- the load imbalance is untouched --
        # while moving each id to an unrelated token, so any speedup here is
        # locality alone. 7919 is prime and the flat length is a power-of-two
        # multiple of k in every padded shape we compile, so the stride is
        # always coprime with it and the map is a bijection.
        #
        # Shuffling *within* each shard, not globally: the MoE is expert
        # parallel, so what the kernel sees is its own device's counts. A
        # global shuffle would both move tokens across devices (changing those
        # counts, which is exactly the variable being held fixed) and cost an
        # all-to-all that would land in the measurement.
        if mesh is None:
            return _shard_local_stride_shuffle(topk_ids)
        spec = P(token_axes, None)
        return jax.shard_map(
            _shard_local_stride_shuffle,
            mesh=mesh,
            in_specs=(spec,),
            out_specs=spec,
            check_vma=False,
        )(topk_ids)

    flat = jax.lax.broadcasted_iota(topk_ids.dtype, (n, k), 0) * k + jax.lax.broadcasted_iota(
        topk_ids.dtype, (n, k), 1
    )
    if mode == "2":
        # 101 is coprime with 288 (and with 304/320), so this is still a
        # bijection on each block of num_physical_experts consecutive slots.
        flat = flat * jnp.asarray(101, topk_ids.dtype)
    return jax.lax.rem(flat, jnp.asarray(num_physical_experts, topk_ids.dtype))


def _router_carries_permutation(config) -> bool:
    """Whether the EPLB map can be baked into the router instead of applied at runtime.

    Two conditions. The map has to be a pure permutation -- with redundant
    experts there are more physical slots than router columns, so the router
    cannot express the mapping at all. And ``n_group`` has to be 1: grouped
    top-k first picks expert *groups*, which are contiguous column ranges, so
    permuting the columns would move experts between groups and change the
    routing decision rather than just relabel it.

    Where it holds, ``create_moe_weights_mapping``'s caller permutes the router
    kernel and bias at load time and the forward pass skips
    ``topk_ids_logical_to_physical`` -- worth 37.7 ms/core/step on a 200K
    prefill, which is more than the rebalancing itself saves.
    """
    from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata

    metadata = get_global_expert_location_metadata()
    if metadata is None:
        return False
    if getattr(config, "n_group", 1) not in (0, 1, None):
        return False
    return int(metadata.num_physical_experts) == int(config.n_routed_experts)


class Glm5NextDecoderLayer(nnx.Module):
    """One layer. Both residual sites go through mHC, so there is no running
    ``residual`` accumulator to thread the way the pre-norm models do."""

    def __init__(
        self,
        config: Glm5NextTextConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.layer_id = layer_id
        self.name = f"Layer_{layer_id:03d}"
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.is_kda = config.is_kda_layer(layer_id)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)

        if self.is_kda:
            self.self_attn = Glm5NextKdaAttention(
                config=config, layer_id=layer_id, mesh=mesh, dtype=dtype
            )
        else:
            self.self_attn = Glm5NextAttention(
                config=config,
                layer_id=layer_id,
                mesh=mesh,
                dtype=dtype,
                use_dsa_sparse=getattr(config, "use_dsa_sparse", False),
            )

        def _hc_site():
            return Glm5NextHyperConnection(
                hidden_size=config.hidden_size,
                hc_mult=config.hc_mult,
                hc_sinkhorn_iters=config.hc_sinkhorn_iters,
                hc_eps=config.hc_eps,
                rms_norm_eps=config.rms_norm_eps,
                mesh=mesh,
                dtype=dtype,
            )

        self.attn_hc = _hc_site()
        self.ffn_hc = _hc_site()
        self.enable_sequence_parallel = getattr(config, "enable_sequence_parallel", False)

        self.is_moe_layer = config.is_sparse_mlp_layer(layer_id)
        if not self.is_moe_layer:
            self.mlp = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mesh=mesh,
                swiglu_limit=self.swiglu_limit,
                dtype=dtype,
            )
            self.moe_gate = None
        else:
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.n_routed_experts,
                enable_expert_bias=True,
                weight_dtype=jnp.float32,
                score_func=config.scoring_func,
            )
            self.topk = TopK(
                topk=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
                routed_scaling_factor=config.routed_scaling_factor,
                layer_id=layer_id,
                mesh=mesh,
            )
            self.router_is_physical = _router_carries_permutation(config)
            # Physical, not logical: with redundant experts FusedEPMoEV2 widens
            # to metadata.num_physical_experts and the debug round-robin has to
            # deal across the same range.
            from sgl_jax.srt.eplb.expert_location import (
                get_global_expert_location_metadata,
            )

            _md = get_global_expert_location_metadata()
            self.num_physical_experts = (
                int(_md.num_physical_experts) if _md is not None else config.n_routed_experts
            )
            moe_backend = getattr(config, "moe_backend", MoEBackend.FUSED_V2)
            if moe_backend != MoEBackend.FUSED_V2.value:
                raise NotImplementedError(
                    f"GLM-5.3 needs the fused_v2 MoE backend (shared expert "
                    f"and swiglu_limit); got {moe_backend!r}"
                )
            num_shared_experts = config.n_shared_experts
            self.mlp = FusedEPMoEV2(
                hidden_size=config.hidden_size,
                num_experts=config.n_routed_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                intermediate_dim=config.moe_intermediate_size,
                mesh=mesh,
                ep_size=getattr(config, "ep_size", 1),
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                renormalize_topk_logits=config.norm_topk_prob,
                routed_scaling_factor=config.routed_scaling_factor,
                use_grouped_topk=config.n_group > 1,
                num_groups=config.n_group,
                top_k_groups=config.topk_group,
                num_shared_experts=num_shared_experts,
                moe_shared_expert_intermediate_size=config.moe_intermediate_size,
                quantization_config=getattr(config, "quantization_config", None),
            )
            self._maybe_add_shared_block_scales(config, num_shared_experts)

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )

    def _maybe_add_shared_block_scales(
        self, config: Glm5NextTextConfig, num_shared_experts: int
    ) -> None:
        """Allocate 2D block scales for the in-kernel shared expert.

        Same checkpoint-compatibility bridge as GLM-5.2: the checkpoint stores
        shared-expert FP8 scales per ``[128, 128]`` block, the fused-v2 shared
        path wants one scale per output channel, so the weights land here and
        get requantized after load.
        """
        quant_config = getattr(config, "quantization_config", None)
        if num_shared_experts <= 0 or quant_config is None:
            return
        if not getattr(quant_config, "is_static_checkpoint", False):
            return
        weight_block_size = getattr(quant_config, "weight_block_size", None)
        if weight_block_size is None:
            return

        block_n, block_k = map(int, weight_block_size)
        shared_intermediate = config.moe_intermediate_size * num_shared_experts
        if (
            config.hidden_size % block_k
            or config.hidden_size % block_n
            or shared_intermediate % block_n
            or shared_intermediate % block_k
        ):
            raise ValueError(
                "GLM-5.3 shared-expert dimensions must be divisible by "
                f"weight_block_size={weight_block_size}"
            )
        for name, (rows, cols) in (
            ("w1_shared_block_scale", (shared_intermediate, config.hidden_size)),
            ("w3_shared_block_scale", (shared_intermediate, config.hidden_size)),
            ("w2_shared_block_scale", (config.hidden_size, shared_intermediate)),
        ):
            setattr(
                self.mlp,
                name,
                nnx.Param(
                    jnp.zeros((rows // block_n, cols // block_k), dtype=jnp.float32),
                    out_sharding=P(None, None),
                ),
            )

    def _token_axes(self, streams: jax.Array):
        """Mesh axes the token dim is split over: ('data','tensor') under SP, else 'data'.

        Derived from ``make_reduce_sharding`` rather than from the flag alone so
        the ``should_scatter`` threshold applies -- a decode step has too few
        tokens to split 16 ways and has to stay on the DP layout.
        """
        return make_reduce_sharding(
            streams, self.mesh, enable_sp=self.enable_sequence_parallel
        ).spec[0]

    def _sp_streams(self, streams: jax.Array) -> NamedSharding:
        """Sharding for the ``[T, hc, D]`` residual streams."""
        return NamedSharding(self.mesh, P(self._token_axes(streams), None, None))

    def _sp_hidden(self, streams: jax.Array) -> NamedSharding:
        """Sharding for a ``[T, D]`` block input/output, matching the streams."""
        return NamedSharding(self.mesh, P(self._token_axes(streams), None))

    @named_scope
    def __call__(
        self,
        streams: jax.Array,
        forward_batch: ForwardBatch,
        memory_pools,
        dispatch_info: ExpertLocationMetadata | None = None,
    ) -> tuple[jax.Array, object, jax.Array | None]:
        # Sequence parallelism. The 4-wide residual stream is the most expensive
        # tensor in the model -- bf16[T, 4, D] is 537 MB at T=16384 -- and under
        # plain TP every one of the 16 cores holds all of it and redoes the same
        # elementwise work on it. Splitting the token axis across ('data',
        # 'tensor') makes the hyper-connection sites, the layernorms and the
        # residual adds 16x cheaper. Collective traffic is unchanged: the
        # all-reduce that used to close each block becomes a reduce-scatter
        # (requested via ``out_sharding`` below) plus the all-gather that feeds
        # the next block's column-parallel matmul.
        #
        # ``sp_streams`` is a no-op after the first layer -- the previous layer
        # already left ``streams`` on this sharding -- and ``make_reduce_sharding``
        # degrades to plain DP when the token count is too small to split, so
        # decode keeps its current layout untouched.
        streams = jax.sharding.reshard(streams, self._sp_streams(streams))
        sp_tokens = self._sp_hidden(streams)
        all_tokens = NamedSharding(self.mesh, P("data", None))

        # ── attention site ──
        residual = streams
        post, comb, hidden_states = self.attn_hc(streams)
        hidden_states = self.input_layernorm(hidden_states)

        # Attention needs every token: the token axis is the sequence, and the
        # parallelism inside the block is over heads.
        hidden_states = jax.sharding.reshard(hidden_states, all_tokens)
        if self.is_kda:
            hidden_states, attn_state = self.self_attn(
                hidden_states,
                forward_batch,
                memory_pools.recurrent_state_pool,
                out_sharding=sp_tokens,
            )
        else:
            hidden_states, attn_state = self.self_attn(
                hidden_states,
                forward_batch,
                memory_pools.token_to_kv_pool,
                out_sharding=sp_tokens,
            )
        streams = _hc_combine(post, comb, hidden_states, residual)

        # ── FFN site ──
        residual = streams
        post, comb, hidden_states = self.ffn_hc(streams)
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe_layer:
            # No all-gather here, unlike the attention site. FusedEPMoEV2 reshards
            # its three token-indexed inputs to P(('data','tensor'), None) before
            # calling the kernel, whose own shard_map in_specs are the same --
            # gathering first would only buy an all-gather plus the slice that
            # undoes it. Routing runs on the SP shard for the same reason: the
            # router is a per-token [D, E] matmul with no cross-token term, so
            # 1/16 of the tokens per core is 1/16 of the work.
            with jax.named_scope("moe"):
                router_logits = self.moe_gate(hidden_states)
                correction_bias = (
                    self.moe_gate.bias.value if self.moe_gate.bias is not None else None
                )
                with jax.named_scope("topk"):
                    topk_weights, topk_ids = self.topk(
                        router_logits,
                        correction_bias,
                        # Router columns were already permuted at load time, so
                        # top-k emits physical ids and there is nothing to map.
                        dispatch_info=None if self.router_is_physical else dispatch_info,
                        routing_sharding=sp_tokens,
                    )
                if _DEBUG_BALANCED_ROUTING:
                    topk_ids = _round_robin_topk_ids(
                        topk_ids,
                        self.num_physical_experts,
                        _DEBUG_BALANCED_ROUTING,
                        mesh=self.mesh,
                        token_axes=self._token_axes(streams),
                    )
                token_valid_mask = forward_batch.get_token_valid_mask(
                    hidden_states.shape[0],
                    out_sharding=NamedSharding(self.mesh, P(self._token_axes(streams))),
                )
                if token_valid_mask is not None:
                    valid = token_valid_mask[:, None]
                    topk_weights = jnp.where(valid, topk_weights, jnp.zeros_like(topk_weights))
                    topk_ids = jnp.where(valid, topk_ids, -1)

                hidden_states = self.mlp(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    out_sharding=sp_tokens,
                    swiglu_limit=self.swiglu_limit,
                    shared_swiglu_limit=self.swiglu_limit,
                )
        else:
            # The dense MLP of the first_k_dense_replace layers *does* need the
            # gather: gate/up are column-parallel over 'tensor', so a token axis
            # also split over 'tensor' would ask one mesh axis to shard two dims
            # of the same matmul.
            hidden_states = jax.sharding.reshard(hidden_states, all_tokens)
            hidden_states = self.mlp(hidden_states, out_sharding=sp_tokens)
            topk_ids = None

        streams = _hc_combine(post, comb, hidden_states, residual)
        return streams, attn_state, topk_ids


class Glm5NextModel(nnx.Module):
    def __init__(
        self,
        config: Glm5NextTextConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.hc_mult = config.hc_mult
        self.vocab_size = config.vocab_size
        self.mesh = mesh

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                Glm5NextDecoderLayer(config=config, mesh=mesh, layer_id=i, dtype=dtype)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="norm",
        )

    def __call__(self, forward_batch: ForwardBatch, memory_pools):
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        # Open the 4-wide residual stream: every stream starts as a copy of the
        # embedding (``inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1)``).
        streams = jnp.broadcast_to(
            hidden_states[:, None, :],
            (hidden_states.shape[0], self.hc_mult, hidden_states.shape[1]),
        )

        layers_kv_fused = []
        layers_recurrent_buffers = []
        layers_conv_buffers = []
        layers_topk_ids = []

        for layer in self.layers:
            streams, attn_state, topk_ids = layer(
                streams,
                forward_batch,
                memory_pools,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            if layer.is_kda:
                rec_buf, conv_buf_list = attn_state
                layers_recurrent_buffers.append(rec_buf)
                layers_conv_buffers.append(conv_buf_list)
            else:
                layers_kv_fused.append(attn_state)
            layers_topk_ids.append(topk_ids)

        # Close sequence parallelism before the head. The layers leave
        # ``streams`` split over ('data','tensor') on the token axis; the logits
        # path indexes individual tokens and expects the plain DP layout. A
        # reshard is a no-op when SP did not engage (decode).
        streams = jax.sharding.reshard(streams, NamedSharding(self.mesh, P("data", None, None)))

        # ``Glm5NextTextHyperHead``: an unweighted mean over the 4 streams.
        hidden_states = jnp.mean(streams, axis=1)
        hidden_states = self.norm(hidden_states)
        return (
            hidden_states,
            layers_kv_fused,
            (layers_recurrent_buffers, layers_conv_buffers),
            layers_topk_ids,
        )


class Glm5NextForConditionalGeneration(nnx.Module):
    """Text-only entry point for GLM-5.3-Flash.

    The class keeps its checkpoint name (``Glm5NextForConditionalGeneration``)
    so ``ModelRegistry`` resolves it, but the vision tower is not built: the
    ``model.visual.*`` checkpoint keys are simply never mapped, and
    ``load_weights_from_safetensors`` only reads keys it has a mapping for.
    """

    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        from sgl_jax.srt.configs.model_config import AttentionArch

        tc = mc.hf_text_config
        mc.head_dim = tc.qk_head_dim
        tc.head_dim = tc.qk_head_dim
        mc.v_head_dim = tc.v_head_dim
        mc.attention_arch = AttentionArch.MLA

        # NoPE: the KV pool sizes itself as
        # ``align_to(kv_lora_rank,128) + align_to(qk_rope_head_dim,128)``, and
        # with _NOPE_PE_WIDTH == 0 that is a flat 512 — matching the kernel,
        # which now skips the pe path entirely. (This used to be forced to 64,
        # costing a padded 128 lanes per token.)
        tc.qk_rope_head_dim = _NOPE_PE_WIDTH

        if mc.quantization_config is not None and (
            mc.quantization_config.is_static_checkpoint
            or getattr(mc.quantization_config, "quantize_on_load", False)
        ):
            # indexer.wk has out_dim=128 == block_size_out (a single N-block);
            # the narrow-N guard would otherwise reject it.
            mc.quantization_config.allow_narrow_n_blockwise = True

            # The KDA short-conv weights are reachable at two paths — the
            # owning attention module (``self_attn.q_conv1d``, which *is* in
            # the checkpoint's modules_to_not_convert) and the RadixLinear-
            # Attention that shares the same object (``self_attn.attn.
            # q_conv1d``, which is not). The quantizer walks both and would
            # convert via the second, leaving the two references pointing at
            # different objects and the weight mapping loading into the wrong
            # one. Ignore entries match by exact suffix, so the bare
            # ``attn.q_conv1d`` form covers every KDA layer.
            ignored = list(mc.quantization_config.ignored_layers or [])
            ignored += ["attn.q_conv1d", "attn.k_conv1d", "attn.v_conv1d"]
            mc.quantization_config.ignored_layers = ignored

    def __init__(
        self,
        config: Glm5NextConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.dtype = dtype
        # The loader hands us the *root* config; everything below the entry
        # class works off text_config. Runtime knobs (ep_size, moe_backend,
        # quantization_config) are attached to the root by ModelRunner /
        # ModelConfig, so forward them onto the text config the submodules see.
        text_config: Glm5NextTextConfig = config.text_config
        for attr in (
            "ep_size",
            "moe_backend",
            "quantization_config",
            "use_dsa_sparse",
            # ModelRunner sets this on the root config only, but the decoder
            # layers read it off text_config.
            "enable_sequence_parallel",
        ):
            value = getattr(config, attr, None)
            if value is not None:
                setattr(text_config, attr, value)
        self.config = text_config
        self.root_config = config

        self.model = Glm5NextModel(text_config, mesh=mesh, dtype=dtype)
        if not getattr(text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(text_config.vocab_size, mesh=mesh)

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.embedding.value
        if getattr(self.config, "tie_word_embeddings", False):
            return embed, embed
        return embed, self.lm_head.embedding.value

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_recurrent_state, layers_topk_ids = self.model(
            forward_batch, memory_pools
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return (
            output,
            {
                "token_to_kv_pool": layers_kv_fused,
                "recurrent_state_pool": layers_recurrent_state,
            },
            True,
            layers_topk_ids,
        )

    # ── weight loading ──

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(self._create_weight_mappings(model_config))

        from sgl_jax.srt.models.glm5_moe import (
            _coarsen_moe_quant_block_k,
            _requantize_glm5_shared_expert,
        )

        # `or 0` rather than a default: serve_pcp.sh exports the var
        # unconditionally, so "not set" reaches us as the empty string.
        coarser_quant_block_k = int(os.environ.get("SGLANG_JAX_MOE_QUANT_BLOCK_K") or 0)

        for layer in self.model.layers:
            if not layer.is_kda:
                layer.self_attn.post_load_weights()
            if isinstance(getattr(layer, "mlp", None), FusedEPMoEV2):
                _requantize_glm5_shared_expert(layer.mlp)
                if coarser_quant_block_k:
                    _coarsen_moe_quant_block_k(layer.mlp, coarser_quant_block_k)
        if coarser_quant_block_k:
            logger.info(
                "Coarsened routed-expert FP8 quant_block_k to %d along K", coarser_quant_block_k
            )
        logger.info("GLM-5.3 weights loaded; absorbed MLA weights folded.")

    def _create_weight_mappings(self, model_config: ModelConfig) -> dict:
        # Text weights live under model.language_model.*, not model.* — the
        # checkpoint is packaged as a VL model. lm_head is bare at the root.
        src = "model.language_model"
        mappings = {
            f"{src}.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            f"{src}.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }
        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint
        is_load_time_quant = bool(
            quant_config is not None and getattr(quant_config, "quantize_on_load", False)
        )

        for layer_id in range(self.config.num_hidden_layers):
            mappings.update(
                self._create_layer_mappings(
                    layer_id,
                    src_prefix=f"{src}.layers.{layer_id}",
                    tgt_prefix=f"model.layers.{layer_id}",
                    is_static_quant=is_static_quant,
                    is_load_time_quant=is_load_time_quant,
                )
            )
        return mappings

    def _create_layer_mappings(
        self,
        layer_id: int,
        *,
        src_prefix: str,
        tgt_prefix: str,
        is_static_quant: bool,
        is_load_time_quant: bool,
    ) -> dict:
        config = self.config
        mappings: dict = {}

        def add_linear(hf: str, tgt: str, sharding_std: tuple, force_unquant: bool = False):
            """HF stores ``[out, in]``.

            Unquantized → ``LinearBase.weight`` ``[in, out]`` (transpose, keep
            kernel_axes). Quantized → ``QuantizedLinear.weight_q`` ``[out, in]``
            (no transpose, swapped sharding) plus the ``weight_scale_inv``
            sidecar when the checkpoint carries one.
            """
            if force_unquant or not (is_static_quant or is_load_time_quant):
                mappings[f"{hf}.weight"] = WeightMapping(
                    target_path=f"{tgt}.weight", sharding=sharding_std, transpose=True
                )
                return
            sharding_q = (sharding_std[1], sharding_std[0])
            mappings[f"{hf}.weight"] = WeightMapping(
                target_path=f"{tgt}.weight_q", sharding=sharding_q, transpose=False
            )
            if is_static_quant:
                mappings[f"{hf}.weight_scale_inv"] = WeightMapping(
                    target_path=f"{tgt}.weight_scale", sharding=(None, None), transpose=False
                )

        def add_norm(hf: str, tgt: str):
            mappings[hf] = WeightMapping(target_path=tgt, sharding=(None,), transpose=False)

        add_norm(f"{src_prefix}.input_layernorm.weight", f"{tgt_prefix}.input_layernorm.scale")
        add_norm(
            f"{src_prefix}.post_attention_layernorm.weight",
            f"{tgt_prefix}.post_attention_layernorm.scale",
        )

        # ── mHC ──
        for hf_site, tgt_site in (("hc_attn", "attn_hc"), ("hc_ffn", "ffn_hc")):
            # fn is [mix, hc*hidden] on disk; we hold it transposed.
            mappings[f"{src_prefix}.{hf_site}_fn"] = WeightMapping(
                target_path=f"{tgt_prefix}.{tgt_site}.fn", sharding=(None, None), transpose=True
            )
            mappings[f"{src_prefix}.{hf_site}_base"] = WeightMapping(
                target_path=f"{tgt_prefix}.{tgt_site}.base", sharding=(None,), transpose=False
            )
            mappings[f"{src_prefix}.{hf_site}_scale"] = WeightMapping(
                target_path=f"{tgt_prefix}.{tgt_site}.scale", sharding=(None,), transpose=False
            )

        # ── attention ──
        ap = f"{src_prefix}.self_attn"
        tp = f"{tgt_prefix}.self_attn"
        if config.is_kda_layer(layer_id):
            # Every KDA tensor is BF16 in this checkpoint — the whole KDA block
            # is listed in modules_to_not_convert.
            for name in ("q_proj", "k_proj", "v_proj", "f_b_proj", "g_b_proj", "b_proj"):
                add_linear(f"{ap}.{name}", f"{tp}.{name}", (None, "tensor"), force_unquant=True)
            for name in ("f_a_proj", "g_a_proj"):
                add_linear(f"{ap}.{name}", f"{tp}.{name}", (None, None), force_unquant=True)
            add_linear(f"{ap}.o_proj", f"{tp}.o_proj", ("tensor", None), force_unquant=True)

            conv_size = config.linear_attn_config["short_conv_kernel_size"]
            projection_size = (
                config.linear_attn_config["num_heads"] * config.linear_attn_config["head_dim"]
            )
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                # HF ships [D, 1, K]; the conv helper wants [D, K].
                mappings[f"{ap}.{name}.weight"] = WeightMapping(
                    target_path=f"{tp}.attn.{name}.weight",
                    sharding=("tensor", None),
                    transpose=False,
                    reshape=(projection_size, conv_size),
                )
            mappings[f"{ap}.o_norm.weight"] = WeightMapping(
                target_path=f"{tp}.o_norm.weight", sharding=(None,), transpose=False
            )
            mappings[f"{ap}.dt_bias"] = WeightMapping(
                target_path=f"{tp}.attn.dt_bias", sharding=("tensor",), transpose=False
            )
            # HF ships a flat [H]; the gate broadcasts against [B, T, H, 1].
            mappings[f"{ap}.A_log"] = WeightMapping(
                target_path=f"{tp}.A_log",
                sharding=(None, None, "tensor", None),
                transpose=False,
                reshape=(1, 1, config.linear_attn_config["num_heads"], 1),
            )
        else:
            add_linear(f"{ap}.q_a_proj", f"{tp}.q_a_proj", (None, None))
            add_norm(f"{ap}.q_a_layernorm.weight", f"{tp}.q_a_layernorm.scale")
            add_linear(f"{ap}.q_b_proj", f"{tp}.q_b_proj", (None, "tensor"))
            add_linear(f"{ap}.kv_a_proj_with_mqa", f"{tp}.kv_a_proj_with_mqa", (None, None))
            add_norm(f"{ap}.kv_a_layernorm.weight", f"{tp}.kv_a_layernorm.scale")
            # kv_b_proj stays BF16 (modules_to_not_convert) and is folded into
            # w_uk/w_uv right after load.
            add_linear(f"{ap}.kv_b_proj", f"{tp}.kv_b_proj", (None, "tensor"), force_unquant=True)
            add_linear(f"{ap}.o_proj", f"{tp}.o_proj", ("tensor", None))

            ix, tix = f"{ap}.indexer", f"{tp}.indexer"
            for name in ("wq_b", "wk", "weights_proj"):
                add_linear(f"{ix}.{name}", f"{tix}.{name}", (None, None), force_unquant=True)
            # Stored as a bare [out, in] tensor, not as a submodule's `.weight`
            # — the reference declares it as an nn.Parameter. We still hold it
            # in a LinearBase, so transpose like any other projection.
            mappings[f"{ix}.index_kpool_compress_gate"] = WeightMapping(
                target_path=f"{tix}.index_kpool_compress_gate.weight",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{ix}.k_norm.weight"] = WeightMapping(
                target_path=f"{tix}.k_norm_weight", sharding=(None,), transpose=False
            )
            mappings[f"{ix}.k_norm.bias"] = WeightMapping(
                target_path=f"{tix}.k_norm_bias", sharding=(None,), transpose=False
            )
            mappings[f"{ix}.index_kpool_compress_ape"] = WeightMapping(
                target_path=f"{tix}.index_kpool_compress_ape",
                sharding=(None, None),
                transpose=False,
            )

        # ── FFN ──
        if not config.is_sparse_mlp_layer(layer_id):
            for name, sharding in (
                ("gate_proj", (None, "tensor")),
                ("up_proj", (None, "tensor")),
                ("down_proj", ("tensor", None)),
            ):
                add_linear(f"{src_prefix}.mlp.{name}", f"{tgt_prefix}.mlp.{name}", sharding)
            return mappings

        from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata

        metadata = get_global_expert_location_metadata()
        phy_to_log = None
        if metadata is not None:
            phy_to_log = np.array(jax.device_get(metadata.physical_to_logical_map))[layer_id]

        # Permute the router to match, so top-k emits physical expert ids
        # directly and the runtime logical->physical gather can be skipped.
        # Measured on a 200K prefill, that gather costs 37.7 ms/core/step --
        # more than the 29.5 ms the rebalancing saves in the MoE kernel, so
        # without this the remap is a net loss. Safe only for a pure
        # permutation (num_physical == num_logical) and only because GLM-5.3
        # has n_group=1: with expert groups, reordering the columns would move
        # experts across group boundaries and change which group top-k picks.
        # See TopK.__call__, which drops dispatch_info under the same condition.
        gate_perm = phy_to_log if _router_carries_permutation(config) else None

        mappings[f"{src_prefix}.mlp.gate.weight"] = WeightMapping(
            target_path=f"{tgt_prefix}.moe_gate.kernel",
            sharding=(None, None),
            transpose=True,
            expert_permutation=gate_perm,
        )
        mappings[f"{src_prefix}.mlp.gate.e_score_correction_bias"] = WeightMapping(
            target_path=f"{tgt_prefix}.moe_gate.bias",
            sharding=(None,),
            expert_permutation=gate_perm,
        )

        moe_backend = getattr(config, "moe_backend", "fused_v2")
        moe_mappings = create_moe_weights_mapping(
            prefix=src_prefix,
            target_prefix=tgt_prefix,
            num_experts=config.n_routed_experts,
            expert_type_names=("gate_proj", "up_proj", "down_proj"),
            moe_backend=moe_backend,
            physical_to_logical_map=phy_to_log,
        )
        if is_static_quant:
            # GLM-5.3 ships block-wise ([128,128]) expert scales. The
            # per-output-channel layout needs a different reshape and scale
            # sharding (see glm5_moe), so refuse it rather than mis-shard.
            layer_quant = getattr(config, "quantization_config", None)
            if getattr(layer_quant, "weight_block_size", None) is None:
                raise NotImplementedError(
                    "GLM-5.3 expects block-wise FP8 expert scales; per-channel "
                    "static quantization is not wired up."
                )
            quantized = {}
            for key, mapping in moe_mappings.items():
                target_param = mapping.target_path[0]
                src_paths = mapping.target_path[1:]
                quantized[key] = WeightMapping(
                    target_path=[target_param] + src_paths,
                    sharding=mapping.sharding,
                    transpose=True,
                    concat_axis=mapping.concat_axis,
                    physical_to_logical_map=mapping.physical_to_logical_map,
                )
                quantized[key + "_scale"] = WeightMapping(
                    target_path=[target_param + "_scale"]
                    + [p.replace(".weight", ".weight_scale_inv") for p in src_paths],
                    # Block-wise: the fused kernels run on the model
                    # (data, tensor) mesh, so the scales follow the routed
                    # weights' expert-axis sharding.
                    sharding=mapping.sharding,
                    transpose=False,
                    concat_axis=mapping.concat_axis,
                    physical_to_logical_map=mapping.physical_to_logical_map,
                )
            moe_mappings = quantized
        mappings.update(moe_mappings)

        num_shared = config.n_shared_experts
        if num_shared > 0:
            sp = f"{src_prefix}.mlp.shared_experts"
            for hf_name, target_name in (
                ("gate_proj", "w1_shared"),
                ("up_proj", "w3_shared"),
                ("down_proj", "w2_shared"),
            ):
                target_path = f"{tgt_prefix}.mlp.{target_name}"
                mappings[f"{sp}.{hf_name}.weight"] = WeightMapping(
                    target_path=target_path, sharding=(None, None), transpose=True
                )
                if is_static_quant:
                    mappings[f"{sp}.{hf_name}.weight_scale_inv"] = WeightMapping(
                        target_path=f"{target_path}_block_scale",
                        sharding=(None, None),
                        transpose=False,
                    )

        return mappings


EntryClass = [Glm5NextForConditionalGeneration]

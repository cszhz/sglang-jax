"""GLM-5.3-Flash (``glm5_next``) config — sgl-jax local definition.

Defined from ``PretrainedConfig`` directly rather than importing a stock
transformers class: the checkpoint declares ``transformers_version 5.16.0``
and production runs 4.57+, so the class may not exist there at all.

Shape of the HF config (``/gcs/models/GLM-5.3-Flash/config.json``)::

    glm5_next
      ├── text_config    (glm5_next_text)   45 layers, hybrid KDA + DSA, MoE
      └── vision_config  (glm5_next_vision) 24-layer ViT

``get_hf_text_config`` already unwraps ``text_config``, so the text
sub-config is what the rest of sgl-jax sees. It mirrors
``_Qwen3_5TextConfig`` closely enough that
``model_runner_kv_cache_mixin._linear_state_params_from_config`` and
friends consume it by duck typing.

Three normalizations happen at construction time, all of them because the
raw HF fields would silently mis-drive existing sgl-jax code:

1. ``head_dim`` ships as ``0``. ``ModelConfig`` reads it with ``getattr``
   (a present-but-zero value defeats the fallback), so it is rewritten to
   ``qk_head_dim`` (256).
2. ``kda_layers`` / ``full_attn_layers`` are **0-indexed** here, whereas
   ``KimiLinearConfig.is_kda_layer`` tests ``(layer_idx + 1) in kda_layers``
   (1-indexed). We never reuse that predicate; layer roles come from the
   explicit ``layer_types`` list and are cross-checked against the two
   index lists so a future off-by-one fails loudly instead of silently
   shifting every layer by one.
3. ``num_nextn_predict_layers`` is ``1`` but the checkpoint carries **zero**
   nextn/mtp tensors (76108 tensors, none matching). It is forced to 0 so
   nothing downstream tries to build an MTP head that has no weights.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

__all__ = ["Glm5NextConfig", "Glm5NextTextConfig", "Glm5NextVisionConfig", "get_glm5_next_config"]

_LINEAR_ATTENTION = "linear_attention"
_FULL_ATTENTION = "deepseek_sparse_attention"


class Glm5NextVisionConfig(PretrainedConfig):
    """ViT sub-config. Declared (not left as a raw dict) so the vision tower
    can be built later without another config pass; the text-only path never
    touches it."""

    model_type = "glm5_next_vision"

    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1024,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        out_hidden_size: int = 4096,
        projection_intermediate_size: int = 10240,
        patch_size: int = 14,
        image_size: int = 448,
        in_channels: int = 3,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        hidden_act: str = "silu",
        swiglu_limit: float = 10.0,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.out_hidden_size = out_hidden_size
        self.projection_intermediate_size = projection_intermediate_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class Glm5NextTextConfig(PretrainedConfig):
    """Text-side sub-config for GLM-5.3-Flash.

    Only fields sgl-jax reads are declared; everything else lands in
    ``**kwargs`` so new HF fields don't break construction.
    """

    model_type = "glm5_next_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        head_dim: int = 0,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # MLA. GLM-5.3 is NoPE: qk_rope_head_dim == 0 and there is no
        # `rope_parameters` block anywhere in the config.
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_head_dim: int = 256,
        qk_nope_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 256,
        mla_use_nope: bool = True,
        # DSA indexer, incl. the k-pooling GLM-5.2 did not have.
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        index_topk: int = 2048,
        index_kpool: int = 4,
        index_kpool_compress: bool = True,
        index_kpool_always_select_tail: bool = True,
        index_share_for_mtp_iteration: bool = True,
        indexer_rope_interleave: bool = True,
        indexer_types: list[str] | None = None,
        # Hybrid schedule. GLM-5.3-Flash ships an explicit `layer_types`; the
        # interval is only the fallback for a bare `Glm5NextTextConfig()`
        # (transformers builds one in `to_diff_dict`).
        full_attention_interval: int = 4,
        layer_types: list[str] | None = None,
        mlp_layer_types: list[str] | None = None,
        linear_attn_config: dict | None = None,
        # MoE.
        moe_intermediate_size: int = 2048,
        n_routed_experts: int = 288,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        first_k_dense_replace: int = 3,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        n_group: int = 1,
        topk_group: int = 1,
        moe_router_dtype: str = "float32",
        router_aux_loss_coef: float = 0.001,
        output_router_logits: bool = False,
        swiglu_limit: float = 10.0,
        # MHC (hyper-connections).
        mhc: bool = True,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # Declared as 1 by HF, but the checkpoint has no nextn weights. See
        # the module docstring; forced to 0 below.
        num_nextn_predict_layers: int = 0,
        pad_token_id: int | None = None,
        eos_token_id: Any = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # MLA dims. `head_dim: 0` on disk; ModelConfig reads it via getattr so
        # the zero would win over the hidden_size//heads fallback and then
        # divide into every attention shape.
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.head_dim = head_dim if head_dim else qk_head_dim
        assert (
            qk_nope_head_dim + qk_rope_head_dim == qk_head_dim
        ), f"qk_nope({qk_nope_head_dim}) + qk_rope({qk_rope_head_dim}) != qk_head_dim({qk_head_dim})"
        # NoPE: there is no rope section at all. Kept explicit (rather than
        # absent) so RoPE builders that probe with getattr see a definite
        # "no rope" instead of falling back to a default theta.
        self.rope_scaling = None
        self.rope_theta = None

        # Indexer.
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        self.index_kpool_compress = index_kpool_compress
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.index_share_for_mtp_iteration = index_share_for_mtp_iteration
        self.indexer_rope_interleave = indexer_rope_interleave
        self.indexer_types = (
            list(indexer_types) if indexer_types is not None else ["full"] * num_hidden_layers
        )

        # Attention schedule. `layer_types` is authoritative (it is the only
        # field guaranteed to be num_hidden_layers long); the two index lists
        # inside linear_attn_config are cross-checked against it.
        self.full_attention_interval = full_attention_interval
        if layer_types is not None:
            assert len(layer_types) == num_hidden_layers, (
                f"layer_types has {len(layer_types)} entries but num_hidden_layers"
                f" is {num_hidden_layers}"
            )
            self.layer_types = list(layer_types)
        elif linear_attn_config is not None:
            kda = set(linear_attn_config["kda_layers"])
            self.layer_types = [
                _LINEAR_ATTENTION if i in kda else _FULL_ATTENTION
                for i in range(num_hidden_layers)
            ]
        else:
            self.layer_types = [
                _FULL_ATTENTION if (i + 1) % full_attention_interval == 0 else _LINEAR_ATTENTION
                for i in range(num_hidden_layers)
            ]

        self.linear_attn_config = linear_attn_config
        if linear_attn_config is not None:
            assert linear_attn_config.get("kda_layers") is not None
            assert linear_attn_config.get("full_attn_layers") is not None
            # 0-indexed here. Cross-check both directions so an index-base
            # mistake surfaces at config construction rather than as garbage
            # logits 45 layers deep.
            assert sorted(linear_attn_config["kda_layers"]) == self.linear_layer_ids, (
                "linear_attn_config.kda_layers disagrees with layer_types; note GLM-5.3"
                " uses 0-indexed layer ids (KimiLinearConfig.is_kda_layer is 1-indexed)"
            )
            assert (
                sorted(linear_attn_config["full_attn_layers"]) == self.full_attention_layer_ids
            ), "linear_attn_config.full_attn_layers disagrees with layer_types"

        # FFN schedule: first_k_dense_replace dense layers, then sparse.
        if mlp_layer_types is not None:
            assert len(mlp_layer_types) == num_hidden_layers
            self.mlp_layer_types = list(mlp_layer_types)
        else:
            self.mlp_layer_types = [
                "dense" if i < first_k_dense_replace else "sparse"
                for i in range(num_hidden_layers)
            ]

        # MoE.
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_router_dtype = moe_router_dtype
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.swiglu_limit = swiglu_limit

        # MHC.
        self.mhc = mhc
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        # The checkpoint ships no nextn tensors; refuse to advertise MTP.
        self.num_nextn_predict_layers = 0
        self.declared_num_nextn_predict_layers = num_nextn_predict_layers

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self) -> bool:
        return True

    @property
    def is_moe(self) -> bool:
        return self.n_routed_experts is not None and self.n_routed_experts > 0

    @property
    def is_linear_attn(self) -> bool:
        return bool(self.linear_layer_ids)

    @property
    def linear_layer_ids(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if str(t).lower() == _LINEAR_ATTENTION]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if str(t).lower() != _LINEAR_ATTENTION]

    def is_kda_layer(self, layer_idx: int) -> bool:
        """0-indexed, unlike ``KimiLinearConfig.is_kda_layer``."""
        return str(self.layer_types[layer_idx]).lower() == _LINEAR_ATTENTION

    def is_sparse_mlp_layer(self, layer_idx: int) -> bool:
        return str(self.mlp_layer_types[layer_idx]).lower() == "sparse"

    @property
    def linear_state_params(self):
        """Sizing block for ``RecurrentStatePool`` over the KDA layers.

        GLM-5.3's KDA is symmetric (q/k/v all ``num_heads x head_dim``), so
        the K-side fields are left to default to the V-side ones.
        """
        from sgl_jax.srt.mem_cache.recurrent_state_pool import (
            LinearRecurrentStateParams,
            recurrent_state_dtype,
        )

        return LinearRecurrentStateParams(
            layers=self.linear_layer_ids,
            num_heads=self.linear_attn_config["num_heads"],
            head_dim=self.linear_attn_config["head_dim"],
            conv_kernel_size=self.linear_attn_config["short_conv_kernel_size"],
            dtype=recurrent_state_dtype(),
        )


class Glm5NextConfig(PretrainedConfig):
    """Root config for GLM-5.3-Flash."""

    model_type = "glm5_next"
    sub_configs = {"text_config": Glm5NextTextConfig, "vision_config": Glm5NextVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    # Disk layout is per-expert (``mlp.experts.<i>.{gate,up,down}_proj``),
    # not pre-fused like Qwen3.5.
    moe_pre_fused: bool = False

    def __init__(
        self,
        text_config: dict | Glm5NextTextConfig | None = None,
        vision_config: dict | Glm5NextVisionConfig | None = None,
        image_token_id: int = 154854,
        video_token_id: int = 154855,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        video_start_token_id: int = 154832,
        video_end_token_id: int = 154833,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if text_config is None:
            text_config = {}
        if isinstance(text_config, dict):
            text_config = Glm5NextTextConfig(**text_config)
        self.text_config = text_config

        if isinstance(vision_config, dict):
            vision_config = Glm5NextVisionConfig(**vision_config)
        self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def get_glm5_next_config(hf_config: Any) -> Glm5NextConfig | None:
    """Return the hf_config if it describes GLM-5.3-Flash, else ``None``.

    Mirrors ``get_kimi_linear_config`` / ``get_qwen3_5_hybrid_config`` so the
    runner dispatches through the same duck-typed hook.
    """
    if getattr(hf_config, "model_type", None) != "glm5_next":
        return None
    if isinstance(hf_config, Glm5NextConfig):
        return hf_config
    config_kwargs = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(vars(hf_config))
    return Glm5NextConfig(**config_kwargs)

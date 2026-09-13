from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
from flax import nnx
from jax.tree_util import register_pytree_node_class

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


@register_pytree_node_class
@dataclass
class AttentionBackendMetadata:
    """Empty pytree base type for per-backend forward metadata.

    Concrete backends (FlashAttention, MLA, ...) subclass this so
    HybridLinearAttnBackendMetadata can type its `full_attn_metadata` field
    without depending on any specific concrete backend.
    """

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()


class AttentionBackend(nnx.Module):
    """The base class of attention backends"""

    # Whether ``ForwardBatch.cache_loc`` has to be uploaded to device.
    #
    # ``cache_loc`` is the token-granular KV slot list, padded to the whole
    # pool: at 200K/bs32 that is s32[6553600] = 26 MB, replicated to every
    # local device on every step. The paged backends never read it on device --
    # they stride it by ``page_size`` on the host in ``get_forward_metadata``
    # and upload only the resulting page table -- so for them the transfer is
    # dead weight that dominates decode. Only backends that index the KV pool
    # per token inside the graph need it; they opt in.
    needs_device_cache_loc: bool = False

    @abstractmethod
    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Init the metadata for a forward pass and return it"""
        raise NotImplementedError()

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        """Run forward on an attention layer."""
        raise NotImplementedError()

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        raise NotImplementedError()

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone paged write of one new KV token per decode sequence.

Normally the MLA v2 kernel lands the new KV in the cache itself, fused into the
block-streaming loop (`start_update_kv_cache`). That only works when a single
rank scans every sequence's last KV block. Once decode attention is split
across tensor ranks that each hold a *full replica* of the MLA cache, the rank
that happens to scan the last block would be the only one to write and the
other fifteen replicas would go stale. Those paths run the attention kernel
with `write_kv_cache=False` and call `write_new_kv` instead.

The cache is ~2.5 GB per layer and the write is ~12 KB, so the only thing that
matters is whether the update happens in place. Measured on v7x with a 2.46 GB
cache, B=12, amortized over 11 chained writes in one jit (`bench_kv_scatter.py`):

    flat.at[loc].set(row)          9.45  ms/layer    a full pass over the array
    B x dynamic_update_slice       0.075 ms/layer    in place, ~6 us per row
    pallas_call + io_aliases       core halt         see below

The scatter is flat in B -- 1 row costs the same as 32 -- which is what a full
pass looks like; XLA will not turn a data-dependent scatter into an in-place
poke. DUS does get lowered in place and is what we use.

The Pallas version is not dead code that was never tried: it worked at grid
size 1 and reproducibly took the core down (`RuntimeUnexpectedCoreHalt`) the
moment the grid had two steps, with the cache aliased in and out under the same
data-dependent index map. It would be worth perhaps 0.8 ms/step at B=12 versus
the ~23 ms/step that splitting decode attention buys, so it is not worth
fighting Mosaic for now. If it is ever revisited, do the page in/out with
explicit `make_async_copy` against an `ANY`-space ref -- the pattern
`start_update_kv_cache` in `kernel.py` already uses -- rather than letting
BlockSpec index maps drive an aliased buffer.
"""

import jax
import jax.numpy as jnp


def new_kv_slots(
    kv_lens: jax.Array,  # [S]
    page_indices: jax.Array,  # [total_pages]
    cu_kv_lens: jax.Array,  # [S + 1], page-aligned token prefix sums
    page_size: int,
) -> jax.Array:
    """Flat cache slot of each decode sequence's new token.

    Decode appends exactly one token, so it lands at `kv_lens[s] - 1`. The
    returned index is into the cache viewed as `[num_pages * page_size, lkv]`,
    which lines up with the `[num_pages, page_size // packing, packing, lkv]`
    layout because that layout is slot-major.
    """
    pos = kv_lens - 1
    page = page_indices[cu_kv_lens[:-1] // page_size + pos // page_size]
    return page * page_size + pos % page_size


def write_new_kv(
    new_kv_c: jax.Array,  # [T, lkv]
    cache_kv: jax.Array,  # [num_pages, page_size // kv_packing, kv_packing, lkv]
    kv_lens: jax.Array,  # [S]
    page_indices: jax.Array,  # [total_pages]
    cu_kv_lens: jax.Array,  # [S + 1]
    num_seqs: int,  # static; rows beyond this are batch padding
) -> jax.Array:
    """Write one new KV token per decode sequence into the paged cache."""
    _, sub_per_page, kv_packing, lkv = cache_kv.shape
    page_size = sub_per_page * kv_packing
    slots = new_kv_slots(kv_lens, page_indices, cu_kv_lens, page_size)
    upd = new_kv_c.reshape(-1, 1, 1, 1, lkv)

    def body(s, cache):
        page, slot = slots[s] // page_size, jnp.remainder(slots[s], page_size)
        return jax.lax.dynamic_update_slice(
            cache,
            jax.lax.dynamic_index_in_dim(upd, s, 0, keepdims=False),
            (page, slot // kv_packing, jnp.remainder(slot, kv_packing), 0),
        )

    # Unrolled by fori_loop rather than by Python so the trace stays O(1) in the
    # batch size; XLA keeps the carry in place, so this is ~6 us per sequence.
    return jax.lax.fori_loop(0, num_seqs, body, cache_kv)

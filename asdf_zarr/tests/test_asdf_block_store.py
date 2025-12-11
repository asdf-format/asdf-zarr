from typing import Any

import zarr
import zarr.buffer
import pytest
import numpy as np

from asdf_zarr.storage import ASDFBlockStore, InternalStore


# We can't use the StoreTests harness for ASDFBlockStore
# since it doesn't support writing (which the test harness requires).


# class TestASDFBlockStore(StoreTests):
#     store_cls = ASDFBlockStore
#     buffer_cls = cpu.Buffer
# 
#     async def set(self, store: ASDFBlockStore, key: str, value: Buffer) -> None:
#         # TODO don't use store methods
#         await store.set(key, value)
# 
#     async def get(self, store: ASDFBlockStore, key: str) -> Buffer:
#         # TODO don't use store methods
#         return await store.get(key)
# 
#     @pytest.fixture()
#     def store_kwargs(self) -> dict[str, Any]:
#         # ctx needs:
#         # - generate_block_key
#         # - get_block_data_callback
#         # chuck_block_map_index (index of block containing the map)
#         # zarray_meta: (dict)
#         # - dimension_separator (optional)
#         # - shape
#         # - chunks
#         # TODO ctx, chunk_block_map_index, zarray_meta
#         return {}
# 
#     def test_store_repr(self, store: ASDFBlockStore) -> None:
#         # TODO fix this
#         assert True
#         #assert str(store) == f"memory://{id(store._store_dict)}"
# 
#     def test_store_supports_writes(self, store: ASDFBlockStore) -> None:
#         assert not store.supports_writes
# 
#     def test_store_supports_listing(self, store: ASDFBlockStore) -> None:
#         assert store.supports_listing


def test_asdf_block_store():
    zarray = {
        "zarr_format": 2,
        "shape": (2, 3),
        "chunks": (1, 1),
        "dtype": "|u1",
        "compressor": None,
        "fill_value": 1,
        "order": "C",
        "filters": None,
    }
    chunk_map = np.full((2, 3), -1, dtype="int32").tobytes()

    class FakeContext:
        def get_block_data_callback(self, index, key):
            if index == 42:
                return lambda: chunk_map
            raise Exception(f"Missing {index}")

        def generate_block_key(self):
            return 1

    ctx = FakeContext()
    store = ASDFBlockStore(ctx, 42, zarray)
    z = zarr.open_array(store, zarr_format=2)
    assert np.all(z[:] == 1)
    assert z.shape == (2, 3)

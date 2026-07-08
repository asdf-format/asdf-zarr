from typing import Any

import zarr
import zarr.buffer
from zarr.abc.store import Store
from zarr.core.buffer import Buffer, cpu
import pytest
import numpy as np

from asdf_zarr.storage import ASDFBlockStore

from zarr.testing import StoreTests


class TestASDFBlockStore(StoreTests):
    store_cls = ASDFBlockStore
    buffer_cls = cpu.Buffer

    async def set(self, store: ASDFBlockStore, key: str, value: Buffer) -> None:
        await store.set(key, value)

    async def get(self, store: ASDFBlockStore, key: str) -> Buffer:
        return await store.get(key)

    @pytest.fixture
    async def store(self, open_kwargs: dict[str, Any]) -> Store:
        store = await self.store_cls.open(**open_kwargs)
        # delete this default key to allow using most of the inherited tests
        await store.delete("zarr.json")
        return store

    @pytest.fixture()
    def store_kwargs(self, tmp_path) -> dict[str, Any]:
        zarray = {
            "zarr_format": 3,
            "shape": (2, 3),
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1, 1)}},
            "data_type": "uint8",
            "fill_value": 1,
            "node_type": "array",
            "chunk_key_encoding": {"name": "default"},
            "codecs": [{"name": "bytes"}],
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
        return {"ctx": ctx, "chunk_block_map_index": 42, "zarray_meta": zarray, "tmp_path": tmp_path}

    def test_store_repr(self, store: ASDFBlockStore) -> None:
        assert str(store).startswith("ASDFBlockStore(")

    def test_store_supports_writes(self, store: ASDFBlockStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: ASDFBlockStore) -> None:
        assert store.supports_listing

    def test_store_eq(self, store, store_kwargs):
        # override the equal test since it does not use store to create
        # the second store to test

        # check self equality
        assert store == store

        # check that 2 store with the same open args are equal
        assert store.__class__(**store_kwargs) == store.__class__(**store_kwargs)

    def test_store_key_exists(self, open_kwargs: dict[str, Any]) -> None:
        thisstore = await self.store_cls.open(**open_kwargs)
        thisttore.delete("fill_value")
        assert not thisstore.exists("fill_value")
        assert thisstore.exists("shape")
        assert thisstore.exists("zarr.json")


async def test_asdf_block_store():
    zarray = {
        "zarr_format": 3,
        "shape": (2, 3),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1, 1)}},
        "data_type": "uint8",
        "fill_value": 1,
        "node_type": "array",
        "chunk_key_encoding": {"name": "default"},
        "codecs": [{"name": "bytes"}],
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
    z = zarr.open_array(store, zarr_format=3)
    assert np.all(z[:] == 1)
    assert z.shape == (2, 3)

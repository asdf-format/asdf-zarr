from typing import Any

from zarr.storage import MemoryStore
from zarr.core.buffer import Buffer, cpu
from zarr.testing import StoreTests

import pytest

from asdf_zarr.storage import WrappedStore


class TestWrappedStore(StoreTests):
    store_cls = WrappedStore
    buffer_cls = cpu.Buffer

    async def set(self, store: WrappedStore, key: str, value: Buffer) -> None:
        await store._wrapped_store.set(key, value)

    async def get(self, store: WrappedStore, key: str) -> Buffer:
        return await store._wrapped_store.get(key)

    @pytest.fixture()
    def store_kwargs(self) -> dict[str, Any]:
        return {"store": MemoryStore()}

    def test_store_repr(self, store: WrappedStore) -> None:
        assert str(store).startswith("WrappedStore(")

    def test_store_supports_writes(self, store: WrappedStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: WrappedStore) -> None:
        assert store.supports_listing

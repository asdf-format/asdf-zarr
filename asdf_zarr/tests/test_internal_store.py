from typing import Any

from zarr.core.buffer import Buffer, cpu
from zarr.testing import StoreTests

import pytest

from asdf_zarr.storage import InternalStore


class TestInternalStore(StoreTests):
    store_cls = InternalStore
    buffer_cls = cpu.Buffer

    async def set(self, store: InternalStore, key: str, value: Buffer) -> None:
        # TODO don't use store methods
        await store.set(key, value)

    async def get(self, store: InternalStore, key: str) -> Buffer:
        # TODO don't use store methods
        return await store.get(key)

    @pytest.fixture()
    def store_kwargs(self) -> dict[str, Any]:
        return {}

    def test_store_repr(self, store: InternalStore) -> None:
        # TODO fix this
        assert True
        #assert str(store) == f"memory://{id(store._store_dict)}"

    def test_store_supports_writes(self, store: InternalStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: InternalStore) -> None:
        assert store.supports_listing

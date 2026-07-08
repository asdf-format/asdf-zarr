import itertools
import json
import math
import tempfile

import numpy
import zarr

from zarr.core.common import concurrent_map
from zarr.core.sync import sync


MISSING_CHUNK = -1


async def _async_iter_to_list(async_iter):
    return [gen async for gen in async_iter]


def async_iter_to_list(async_iter):
    return sync(_async_iter_to_list(async_iter))


def _iter_chunk_keys(zarray, only_initialized=False):
    """Using zarray metadata iterate over chunk keys"""
    if only_initialized:
        for k in async_iter_to_list(zarray.store.list()):
            if k in (".zarray", ".zattrs", "zarr.json"):
                continue
            yield k
        return
    # load meta
    zarray_meta = zarray.metadata.to_dict()
    dimension_separator = zarray_meta.get("dimension_separator", "/")

    # make blocks and map them to the internal kv store
    # compute number of chunks (across all axes)
    chunk_counts = [
        math.ceil(s / c)
        for (s, c) in zip(zarray_meta["shape"], zarray_meta["chunk_grid"]["configuration"]["chunk_shape"])
    ]

    # iterate over all chunk keys
    chunk_iter = itertools.product(*[range(c) for c in chunk_counts])
    for c in chunk_iter:
        key = dimension_separator.join([str(i) for i in c])
        yield key


def _generate_chunk_data_callback(zarray, chunk_key):
    def chunk_data_callback(zarray=zarray, chunk_key=chunk_key):
        return numpy.frombuffer(sync(zarray.store.get(chunk_key)).to_bytes(), dtype="uint8")

    return chunk_data_callback


def _generate_chunk_map_callback(zarray, chunk_key_block_index_map):
    # make an array
    def chunk_map_callback(zarray=zarray, chunk_key_block_index_map=chunk_key_block_index_map):
        chunk_map = numpy.zeros(zarray.cdata_shape, dtype="int32")
        chunk_map[:] = MISSING_CHUNK  # set all as uninitialized
        zarray_meta = zarray.metadata.to_dict()
        dimension_separator = zarray_meta.get("dimension_separator", "/")
        for k in _iter_chunk_keys(zarray, only_initialized=True):
            index = chunk_key_block_index_map[k]
            # zarr format v3 uses 'c' and the separator as a prefix for coordinates so that needs to be stripped if there
            coords = k.split(dimension_separator)
            if coords[0].lower() == "c":
                coords = coords[1:]
            coords = tuple([int(sk) for sk in coords])
            chunk_map[coords] = index
        return chunk_map

    return chunk_map_callback


def to_internal(zarray):
    if isinstance(zarray.store, WrappedStore):
        return zarray
    # make a new internal store based off an existing store
    internal_store = WrappedStore(zarray.store)
    return zarr.open(internal_store)


class WrappedStore(zarr.abc.store.Store):
    def __init__(self, store=None, read_only=False):
        super().__init__()
        self._wrapped_store = store
        self._read_only = read_only

    @property
    def supports_writes(self):
        return self._wrapped_store.supports_writes

    @property
    def supports_deletes(self):
        return self._wrapped_store.supports_deletes

    @property
    def supports_listing(self):
        return self._wrapped_store.supports_listing

    @property
    def supports_partial_writes(self):
        return self._wrapped_store.supports_partial_writes

    @property
    def read_only(self):
        return self._read_only

    def __eq__(self, other):
        return isinstance(other, WrappedStore) and self._wrapped_store == other._wrapped_store

    def __repr__(self):
        return f"WrappedStore({self._wrapped_store.__class__.__name__}, '{self._wrapped_store}')"

    async def set(self, key, value):
        if self.read_only:
            raise ValueError("store was opened in read-only mode and does not support writing")
        return await self._wrapped_store.set(key, value)

    async def get(self, key, prototype=None, byte_range=None):
        return await self._wrapped_store.get(key, prototype, byte_range)

    async def delete(self, key):
        if self.read_only:
            raise ValueError("store was opened in read-only mode and does not support writing")
        return await self._wrapped_store.delete(key)

    async def exists(self, key):
        return await self._wrapped_store.exists(key)

    async def get_partial_values(self, prototype=None, key_ranges=None):
        return await self._wrapped_store.get_partial_values(prototype, key_ranges)

    async def list(self):
        async for key in self._wrapped_store.list():
            yield key

    async def list_dir(self, prefix):
        async for key in self._wrapped_store.list_dir(prefix):
            yield key

    async def list_prefix(self, prefix):
        async for key in self._wrapped_store.list_prefix(prefix):
            yield key


class ASDFBlockStore(zarr.abc.store.Store):
    supports_listing = True
    supports_partial_writes = False

    def __init__(self, ctx, chunk_block_map_index, zarray_meta, tmp_path=None, read_only=False):
        super().__init__()

        if tmp_path is None:
            self._tmp_dir = tempfile.TemporaryDirectory()
            tmp_path = self._tmp_dir.name
        else:
            tmp_path = str(tmp_path)
        self._tmp_store = zarr.storage.LocalStore(tmp_path)
        self._zarray_meta = zarr.buffer.cpu.Buffer.from_bytes(json.dumps(zarray_meta).encode("ascii"))

        self._deleted_keys = set()
        self._read_only = read_only

        # the chunk_block_map contains block indices
        # organized in an array shaped like the chunks
        # so for a zarray with 4 x 5 chunks (dimension 1
        # split into 4 chunks) the chunk_block_map will be
        # 4 x 5
        cdata_shape = tuple(
            math.ceil(s / c)
            for s, c in zip(zarray_meta["shape"], zarray_meta["chunk_grid"]["configuration"]["chunk_shape"])
        )
        self._chunk_block_map_asdf_key = ctx.generate_block_key()
        self._chunk_block_map = numpy.frombuffer(
            ctx.get_block_data_callback(chunk_block_map_index, self._chunk_block_map_asdf_key)(), dtype="int32"
        ).reshape(cdata_shape)

        self._chunk_block_map_asdf_key = None

        # reorganize the map into a set and claim the block indices
        self._chunk_callbacks = {}
        self._chunk_asdf_keys = {}
        _sep = zarray_meta.get("dimension_separator", "/")
        for coord in numpy.transpose(numpy.nonzero(self._chunk_block_map != MISSING_CHUNK)):
            coord = tuple(coord)
            block_index = int(self._chunk_block_map[coord])
            chunk_key = "c/" + _sep.join((str(c) for c in tuple(coord)))
            asdf_key = ctx.generate_block_key()
            self._chunk_asdf_keys[chunk_key] = asdf_key
            self._chunk_callbacks[chunk_key] = ctx.get_block_data_callback(block_index, asdf_key)

    @property
    def read_only(self):
        return self._read_only

    @property
    def supports_writes(self):
        return not self.read_only

    @property
    def supports_deletes(self):
        return not self.read_only

    def __eq__(self, other):
        if not isinstance(other, ASDFBlockStore):
            return False
        if self._tmp_store != other._tmp_store:
            return False
        if self._deleted_keys != other._deleted_keys:
            return False
        if self._read_only != other._read_only:
            return False
        if self._zarray_meta != other._zarray_meta:
            return False
        if self._chunk_callbacks != other._chunk_callbacks:
            return False
        return True

    def __repr__(self):
        return f"ASDFBlockStore({id(self)})"

    async def set(self, key, value):
        if not self.supports_writes:
            raise ValueError("store was opened in read-only mode and does not support writing")
        if key in self._deleted_keys:
            self._deleted_keys.remove(key)
        return await self._tmp_store.set(key, value)

    async def get(self, key, prototype=None, byte_range=None):
        # first check deleted_keys
        if key in self._deleted_keys:
            return None

        # then tmp_store
        if await self._tmp_store.exists(key):
            return await self._tmp_store.get(key, prototype, byte_range)

        if key == "zarr.json":
            return self._zarray_meta

        # then blocks
        if key not in self._chunk_callbacks:
            return None
        data = self._chunk_callbacks[key]()
        return zarr.buffer.cpu.Buffer.from_bytes(data)

    async def delete(self, key):
        if not self.supports_deletes:
            raise ValueError("store was opened in read-only mode and does not support writing")
        self._deleted_keys.add(key)

    async def exists(self, key):
        # first check deleted keys
        if key in self._deleted_keys:
            return False

        # then tmp_store
        if await self._tmp_store.exists(key):
            return True

        if key == "zarr.json":
            return True

        # then blocks
        return key in self._chunk_callbacks

    async def get_partial_values(self, prototype=None, key_ranges=None):
        # All the key-ranges arguments goes with the same prototype
        async def _get(key: str, byte_range):
            return await self.get(key, prototype=prototype, byte_range=byte_range)

        return await concurrent_map(key_ranges, _get, limit=None)

    async def list(self):
        reported = set()
        async for key in self._tmp_store.list():
            if key not in self._deleted_keys and key not in reported:
                reported.add(key)
                yield key
        if "zarr.json" not in reported and "zarr.json" not in self._deleted_keys:
            yield "zarr.json"
        for key in self._chunk_callbacks:
            if key not in self._deleted_keys and key not in reported:
                reported.add(key)
                yield key

    async def list_dir(self, prefix):
        if prefix.endswith("/"):
            dir_prefix = prefix
        else:
            dir_prefix = prefix + "/"
        async for key in self.list():
            if key.startswith(dir_prefix):
                key = key.removeprefix(dir_prefix)
                if "/" in key:
                    yield key.split("/")[0]
                else:
                    yield key

    async def list_prefix(self, prefix):
        async for key in self.list():
            if key.startswith(prefix):
                yield key

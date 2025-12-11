import asyncio
import itertools
import json
import math
import tempfile

import asdf
import numpy
import zarr


MISSING_CHUNK = -1


async def _async_iter_to_list(async_iter):
    return [gen async for gen in async_iter]


def async_iter_to_list(async_iter):
    return asyncio.run(_async_iter_to_list(async_iter))


def _iter_chunk_keys(zarray, only_initialized=False):
    """Using zarray metadata iterate over chunk keys"""
    if only_initialized:
        for k in async_iter_to_list(zarray.store.list()):
            if k in (".zarray", ".zattrs"):
                continue
            yield k
        return
    # load meta
    zarray_meta = zarray.metadata.to_dict()
    dimension_separator = zarray_meta.get("dimension_separator", ".")

    # make blocks and map them to the internal kv store
    # compute number of chunks (across all axes)
    chunk_counts = [math.ceil(s / c) for (s, c) in zip(zarray_meta["shape"], zarray_meta["chunks"])]

    # iterate over all chunk keys
    chunk_iter = itertools.product(*[range(c) for c in chunk_counts])
    for c in chunk_iter:
        key = dimension_separator.join([str(i) for i in c])
        yield key


def _generate_chunk_data_callback(zarray, chunk_key):
    def chunk_data_callback(zarray=zarray, chunk_key=chunk_key):
        return numpy.frombuffer(asyncio.run(zarray.store.get(chunk_key)).to_bytes(), dtype="uint8")

    return chunk_data_callback


def _generate_chunk_map_callback(zarray, chunk_key_block_index_map):
    # make an array
    def chunk_map_callback(zarray=zarray, chunk_key_block_index_map=chunk_key_block_index_map):
        chunk_map = numpy.zeros(zarray.cdata_shape, dtype="int32")
        chunk_map[:] = MISSING_CHUNK  # set all as uninitialized
        zarray_meta = zarray.metadata.to_dict()
        dimension_separator = zarray_meta.get("dimension_separator", ".")
        for k in _iter_chunk_keys(zarray, only_initialized=True):
            index = chunk_key_block_index_map[k]
            coords = tuple([int(sk) for sk in k.split(dimension_separator)])
            chunk_map[coords] = index
        return chunk_map

    return chunk_map_callback


def to_internal(zarray):
    if isinstance(zarray.store, InternalStore):
        return zarray
    # make a new internal store based off an existing store
    internal_store = ConvertedInternalStore(zarray.store)
    return zarr.open(zarray.store, chunk_store=internal_store)


#'__eq__', 'get_partial_values', 'list', 'list_dir', 'list_prefix'
class InternalStore(zarr.abc.store.Store):
    supports_deletes = True
    supports_listing = True
    supports_partial_writes = False
    supports_writes = True

    def __init__(self, read_only=False):
        super().__init__()
        # TODO support read_only? a requirement of zarr?
        self._read_only = read_only
        self._tmp_store_ = None
        self._deleted_keys = set()

    @property
    def read_only(self):
        return self._read_only

    def __eq__(self, other):
        # TODO make this robust
        return id(self) == id(other)

    @property
    def _tmp_store(self):
        if self._tmp_store_ is None:
            self._tmp_dir = tempfile.TemporaryDirectory()
            self._tmp_store_ = zarr.storage.LocalStore(self._tmp_dir.name, read_only=self.read_only)
        return self._tmp_store_

    async def set(self, key, value):
        if self.read_only:
            raise ValueError("store was opened in read-only mode and does not support writing")
        if key in self._deleted_keys:
            self._deleted_keys.remove(key)
        return await self._tmp_store.set(key, value)

    async def get(self, key, prototype=None, byte_range=None):
        if key in self._deleted_keys or self._tmp_store_ is None:
            raise FileNotFoundError(f"{key}")
        return await self._tmp_store.get(key, prototype, byte_range)

    async def delete(self, key):
        if self.read_only:
            raise ValueError("store was opened in read-only mode and does not support writing")
        self._deleted_keys.add(key)

    async def exists(self, key):
        if key in self._deleted_keys:
            return False
        return await self._tmp_store.exists(key)

    async def get_partial_values(self, prototype=None, key_ranges=None):
        # TODO handle deleted keys
        return await self._tmp_store.get_partial_values(prototype, key_ranges)

    async def list(self):
        async for key in self._tmp_store.list():
            if key not in self._deleted_keys:
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


class ConvertedInternalStore(InternalStore):
    def __init__(self, existing):
        super().__init__()
        self._existing_store = existing
        self._chunk_asdf_keys = {}
        self._chunk_block_map_asdf_key = None

    def __getitem__(self, key):
        if key in self._deleted_keys:
            raise KeyError(f"{key}")
        try:
            return super().__getitem__(key)
        except KeyError:
            pass
        return self._existing_store.__getitem__(key)

    def __iter__(self):
        keys = set(super().__iter__())
        keys = keys.union(set(self._existing_store.__iter__()))
        return iter(keys.difference(self._deleted_keys))


class ReadInternalStore(InternalStore):
    def __init__(self, ctx, chunk_block_map_index, zarray_meta):
        super().__init__()

        self._sep = zarray_meta.get("dimension_separator", ".")

        # the chunk_block_map contains block indices
        # organized in an array shaped like the chunks
        # so for a zarray with 4 x 5 chunks (dimension 1
        # split into 4 chunks) the chunk_block_map will be
        # 4 x 5
        cdata_shape = tuple(math.ceil(s / c) for s, c in zip(zarray_meta["shape"], zarray_meta["chunks"]))
        self._chunk_block_map_asdf_key = ctx.generate_block_key()
        self._chunk_block_map = numpy.frombuffer(
            ctx.get_block_data_callback(chunk_block_map_index, self._chunk_block_map_asdf_key)(), dtype="int32"
        ).reshape(cdata_shape)

        self._chunk_block_map_asdf_key = None

        # reorganize the map into a set and claim the block indices
        self._chunk_callbacks = {}
        self._chunk_asdf_keys = {}
        for coord in numpy.transpose(numpy.nonzero(self._chunk_block_map != MISSING_CHUNK)):
            coord = tuple(coord)
            block_index = int(self._chunk_block_map[coord])
            chunk_key = self._sep.join((str(c) for c in tuple(coord)))
            asdf_key = ctx.generate_block_key()
            self._chunk_asdf_keys[chunk_key] = asdf_key
            self._chunk_callbacks[chunk_key] = ctx.get_block_data_callback(block_index, asdf_key)

    def __getstate__(self):
        state = {}
        state["_sep"] = self._sep
        if hasattr(self, "_chunk_info"):
            # handle instance that was already pickled and unpickled
            state["_chunk_info"] = self._chunk_info
        else:
            # and instance that was not yet pickled

            # for each callback, get the file uri and block offset
            def _callback_info(cb):
                return {
                    "offset": cb(_attr="offset"),
                    "uri": cb(_attr="_fd")().uri,
                }

            state["_chunk_info"] = {k: _callback_info(self._chunk_callbacks[k]) for k in self._chunk_callbacks}
        return state

    def __setstate__(self, state):
        self._sep = state["_sep"]

        def _to_callback(info):
            def cb():
                with asdf.generic_io.get_file(info["uri"], mode="r") as gf:
                    return asdf._block.io.read_block(gf, info["offset"])[-1]

            return cb

        self._chunk_info = state["_chunk_info"]
        self._chunk_callbacks = {k: _to_callback(self._chunk_info[k]) for k in self._chunk_info}
        # as __init__ will not be called on self, set up attributed expected
        # due to the parent InternalStore class
        self._tmp_store_ = None
        self._deleted_keys = set()

    def _sep_key(self, key):
        if self._sep is None:
            return key
        return key.split(self._sep)

    def _coords(self, key):
        return tuple([int(sk) for sk in self._sep_key(key)])

    def __getitem__(self, key):
        if key in self._deleted_keys:
            raise KeyError(f"{key}")
        try:
            return super().__getitem__(key)
        except KeyError:
            pass
        if key not in self._chunk_callbacks:
            raise KeyError(f"{key}")
        return self._chunk_callbacks.get(key, None)()

    def __iter__(self):
        keys = set(super().__iter__())
        keys = keys.union(set(self._chunk_callbacks))
        return iter(keys.difference(self._deleted_keys))

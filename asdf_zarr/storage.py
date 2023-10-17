import itertools
import json
import math

import asdf
import numpy
import zarr


MISSING_CHUNK = -1


def _iter_chunk_keys(zarray, only_initialized=False):
    """Using zarray metadata iterate over chunk keys"""
    if only_initialized:
        for k in zarr.storage.listdir(zarray.chunk_store):
            if k == ".zarray":
                continue
            yield k
        return
    # load meta
    zarray_meta = json.loads(zarray.store[".zarray"])
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
        return numpy.frombuffer(zarray.chunk_store.get(chunk_key), dtype="uint8")

    return chunk_data_callback


def _generate_chunk_map_callback(zarray, chunk_key_block_index_map):
    # make an array
    def chunk_map_callback(zarray=zarray, chunk_key_block_index_map=chunk_key_block_index_map):
        chunk_map = numpy.zeros(zarray.cdata_shape, dtype="int32")
        chunk_map[:] = MISSING_CHUNK  # set all as uninitialized
        zarray_meta = json.loads(zarray.store[".zarray"])
        dimension_separator = zarray_meta.get("dimension_separator", ".")
        for k in _iter_chunk_keys(zarray, only_initialized=True):
            index = chunk_key_block_index_map[k]
            coords = tuple([int(sk) for sk in k.split(dimension_separator)])
            chunk_map[coords] = index
        return chunk_map

    return chunk_map_callback


def to_internal(zarray):
    if isinstance(zarray.chunk_store, InternalStore):
        return zarray
    # make a new internal store based off an existing store
    internal_store = ConvertedInternalStore(zarray.chunk_store or zarray.store)
    return zarr.open(zarray.store, chunk_store=internal_store)


class InternalStore(zarr.storage.Store):
    def __init__(self):
        super().__init__()
        self._tmp_store_ = None
        self._deleted_keys = set()

    @property
    def _tmp_store(self):
        if self._tmp_store_ is None:
            # TODO options to control where TempStore is stored
            self._tmp_store_ = zarr.storage.TempStore()
        return self._tmp_store_

    def __setitem__(self, key, value):
        if key in self._deleted_keys:
            self._deleted_keys.remove(key)
        self._tmp_store[key] = value

    def __delitem__(self, key):
        self._deleted_keys.add(key)

    def __len__(self):
        return len(set(self.__iter__()))

    def listdir(self, path):
        if path:
            raise NotImplementedError("path argument not supported by InternalStore.listdir")
        return list(self.__iter__())

    def __getitem__(self, key):
        if key in self._deleted_keys or self._tmp_store_ is None:
            raise KeyError(f"{key}")
        return self._tmp_store.__getitem__(key)

    def __iter__(self):
        keys = set()
        if self._tmp_store_ is not None:
            keys = keys.union(set(self._tmp_store.__iter__()))
        return iter(keys.difference(self._deleted_keys))


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

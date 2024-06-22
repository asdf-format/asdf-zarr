import json

import asdf

import zarr

from . import util
from . import storage

# TODO convert imports to local to avoid imports on extension loading


class ZarrConverter(asdf.extension.Converter):
    tags = ["asdf://asdf-format.org/zarr/tags/zarr-*"]
    types = ["zarr.core.Array"]

    def to_yaml_tree(self, obj, tag, ctx):
        chunk_store = obj.chunk_store or obj.store
        # these storage types require conversion to an internal store so make it the default
        if isinstance(chunk_store, (zarr.storage.KVStore, zarr.storage.MemoryStore, zarr.storage.TempStore)):
            chunk_store = storage.ConvertedInternalStore(chunk_store)
        if isinstance(chunk_store, storage.InternalStore):
            # TODO should we enforce no zarr compression here?
            # include data from this zarr array in the asdf file
            meta = json.loads(obj.store[".zarray"])
            obj_dict = {}

            # include the meta data in the tree
            obj_dict[".zarray"] = meta

            # update callbacks
            chunk_key_block_index_map = {}
            for chunk_key in storage._iter_chunk_keys(obj, only_initialized=True):
                data_callback = storage._generate_chunk_data_callback(obj, chunk_key)
                asdf_key = chunk_store._chunk_asdf_keys.get(chunk_key, ctx.generate_block_key())
                block_index = ctx.find_available_block_index(data_callback, asdf_key)
                chunk_key_block_index_map[chunk_key] = block_index
            asdf_key = chunk_store._chunk_block_map_asdf_key
            if asdf_key is None:
                asdf_key = ctx.generate_block_key()
            obj_dict["chunk_block_map"] = ctx.find_available_block_index(
                storage._generate_chunk_map_callback(obj, chunk_key_block_index_map), asdf_key
            )
            return obj_dict

        obj_dict = {}
        if obj.store is not chunk_store:
            # encode meta store
            obj_dict["meta_store"] = util.encode_storage(obj.store)
        obj_dict["store"] = util.encode_storage(chunk_store)
        # TODO mode, version, path_str?
        return obj_dict

    def from_yaml_tree(self, node, tag, ctx):
        import zarr

        from . import util
        from . import storage

        if ".zarray" in node and "chunk_block_map" in node:
            # this is an internally stored zarr array
            # TODO should we enforce no zarr compression here?

            # load the meta data into memory
            store = zarr.storage.KVStore({".zarray": json.dumps(node[".zarray"])})

            # setup an InternalStore to read block data (when requested)
            zarray_meta = node[".zarray"]
            chunk_block_map_index = node["chunk_block_map"]

            chunk_store = storage.ReadInternalStore(ctx, chunk_block_map_index, zarray_meta)

            # TODO read/write mode here
            obj = zarr.open_array(store=store, chunk_store=chunk_store)
            return obj

        chunk_store = util.decode_storage(node["store"])
        if "meta_store" in node:
            # separate meta and chunk stores
            store = util.decode_storage(node["meta_store"])
        else:
            store = chunk_store
        # TODO mode, version, path_str?
        obj = zarr.open(store=store, chunk_store=chunk_store)
        return obj

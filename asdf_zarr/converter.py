import asyncio
import json

import asdf
import zarr
import zarr.buffer

from . import util
from . import storage


class ZarrConverter(asdf.extension.Converter):
    tags = ["asdf://asdf-format.org/zarr/tags/zarr-*"]
    types = ["zarr.core.array.Array"]

    def to_yaml_tree(self, obj, tag, ctx):
        chunk_store = obj.store
        # these storage types require conversion to an internal store so make it the default
        if isinstance(chunk_store, (zarr.storage.MemoryStore, storage.ASDFBlockStore)):
            chunk_store = storage.WrappedStore(chunk_store)
        if isinstance(chunk_store, storage.WrappedStore):
            # TODO should we enforce no zarr compression here?
            # include data from this zarr array in the asdf file
            meta = obj.metadata.to_dict()
            obj_dict = {}

            # include the meta data in the tree
            obj_dict["zarr.json"] = meta

            # update callbacks
            chunk_key_block_index_map = {}
            for chunk_key in storage._iter_chunk_keys(obj, only_initialized=True):
                data_callback = storage._generate_chunk_data_callback(obj, chunk_key)
                asdf_key = getattr(chunk_store, "_chunk_asdf_keys", {}).get(chunk_key, ctx.generate_block_key())
                block_index = ctx.find_available_block_index(data_callback, asdf_key)
                chunk_key_block_index_map[chunk_key] = block_index
            asdf_key = getattr(chunk_store, "_chunk_block_map_asdf_key", None)
            if asdf_key is None:
                asdf_key = ctx.generate_block_key()
            obj_dict["chunk_block_map"] = ctx.find_available_block_index(
                storage._generate_chunk_map_callback(obj, chunk_key_block_index_map), asdf_key
            )
            return obj_dict

        obj_dict = {}
        obj_dict["store"] = util.encode_storage(chunk_store)
        return obj_dict

    def from_yaml_tree(self, node, tag, ctx):
        if "zarr.json" in node and "chunk_block_map" in node:
            # this is an internally stored zarr array
            # setup an ASDFBlockStore to read block data (when requested)
            zarray_meta = node["zarr.json"]
            chunk_block_map_index = node["chunk_block_map"]

            store = storage.ASDFBlockStore(ctx, chunk_block_map_index, zarray_meta)

            # TODO read/write mode here
            obj = zarr.open_array(store=store, zarr_format=2)
            return obj

        store = util.decode_storage(node["store"])
        obj = zarr.open_array(store=store, zarr_format=2)
        return obj

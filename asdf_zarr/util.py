import copy

import fsspec
import zarr.storage
from zarr.storage import DirectoryStore, FSStore, KVStore, NestedDirectoryStore, TempStore


def encode_storage(store):
    """
    Convert a zarr.storage.Store subclass as a dictionary that can
    be stored in an ASDF tree and is sufficient to produce a functionally
    identical store (see `decode_storage` to load obj_dict).

    Parameters
    ----------
    store : zarr.storage.Store

    Returns
    -------
    obj_dict : dictionary encoding
    """
    obj_dict = {"type_string": store.__class__.__name__}
    if isinstance(store, (DirectoryStore, NestedDirectoryStore)) and not isinstance(store, TempStore):
        # dimension separator is _dimension separator and should be
        # read from the zarray itself, not the store
        obj_dict["normalize_keys"] = store.normalize_keys
        obj_dict["path"] = store.path
    elif isinstance(store, FSStore):
        obj_dict["normalize_keys"] = store.normalize_keys
        # store.path path within the filesystem
        obj_dict["path"] = store.path
        # store.mode access mode
        obj_dict["mode"] = store.mode
        # store.fs.to_json to get full filesystem (see fsspec.AbstractFileSystem.from_json)
        obj_dict["fs"] = store.fs.to_json()
    elif isinstance(store, KVStore):
        obj_dict["map"] = {k: store[k] for k in store}
    else:
        raise NotImplementedError(f"zarr.storage.Store subclass {store.__class__} not supported")
    return obj_dict


def decode_storage(obj_dict):  # TODO needs kwargs for dimension sep?
    """
    Convert an dict containing information about a zarr.storage.Store
    into a Store instance (see `encode_storage` to produce obj_dict).

    Parameters
    ----------
    obj_dict : dictionary encoding

    Returns
    -------
    store : zarr.storage.Store
    """
    kwargs = copy.deepcopy(obj_dict)
    args = []
    type_string = kwargs.pop("type_string")
    if not hasattr(zarr.storage, type_string):
        raise NotImplementedError(f"zarr.storage.Store subclass {type_string} not supported")
    if "fs" in kwargs and type_string == "FSStore":
        kwargs["fs"] = fsspec.AbstractFileSystem.from_json(kwargs["fs"])
        args.append(kwargs.pop("path"))
    elif "map" in kwargs and type_string == "KVStore":
        args.append(kwargs.pop("map"))
    return getattr(zarr.storage, type_string)(*args, **kwargs)

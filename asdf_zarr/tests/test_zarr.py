from collections import UserDict
import itertools

import asdf
import asdf_zarr
import asdf_zarr.storage
import numpy
import pytest
import zarr
from zarr.storage import DirectoryStore, KVStore, MemoryStore, NestedDirectoryStore, TempStore


def create_zarray(shape=None, chunks=None, dtype="f8", store=None, chunk_store=None):
    if shape is None:
        shape = (6, 9)
    if chunks is None:
        chunks = [max(1, d // 3) for d in shape]
    arr = zarr.creation.create(
        (6, 9), store=store, chunk_store=chunk_store, chunks=chunks, dtype=dtype, compressor=None
    )
    for chunk_index in itertools.product(*[range(c) for c in arr.cdata_shape]):
        inds = []
        for i, c in zip(chunk_index, arr.chunks):
            inds.append(slice(i * c, (i + 1) * c))
        arr[tuple(inds)] = i
    return arr


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
@pytest.mark.parametrize("compression", ["input", "zlib"])
@pytest.mark.parametrize("store_type", [DirectoryStore, KVStore, MemoryStore, NestedDirectoryStore, TempStore])
@pytest.mark.parametrize("to_internal", [True, False])
@pytest.mark.parametrize("meta_store", [True, False])
def test_write_to(tmp_path, copy_arrays, lazy_load, compression, store_type, to_internal, meta_store):
    if store_type in (DirectoryStore, NestedDirectoryStore):
        store1 = store_type(tmp_path / "zarr_array_1")
        store2 = store_type(tmp_path / "zarr_array_2")
    elif store_type is KVStore:
        store1 = store_type({})
        store2 = store_type({})
    else:
        store1 = store_type()
        store2 = store_type()

    # should meta be in a different store?
    if meta_store:
        chunk_store1 = store1
        store1 = KVStore({})
        chunk_store2 = store2
        store2 = KVStore({})
    else:
        chunk_store1 = None
        chunk_store2 = None

    arr1 = create_zarray(store=store1, chunk_store=chunk_store1)
    arr2 = create_zarray(store=store2, chunk_store=chunk_store2)

    arr2[:] = arr2[:] * -2
    if to_internal:
        arr1 = asdf_zarr.storage.to_internal(arr1)
        arr2 = asdf_zarr.storage.to_internal(arr2)
    tree = {"arr1": arr1, "arr2": arr2}

    fn = tmp_path / "test.asdf"
    af = asdf.AsdfFile(tree)
    af.write_to(fn, all_array_compression=compression)

    with asdf.open(fn, mode="r", copy_arrays=copy_arrays, lazy_load=lazy_load) as af:
        for n, a in (("arr1", arr1), ("arr2", arr2)):
            assert isinstance(af[n], zarr.core.Array)
            if to_internal or store_type in (KVStore, MemoryStore, TempStore):
                assert isinstance(af[n].chunk_store, asdf_zarr.storage.InternalStore)
            else:
                assert isinstance(af[n].chunk_store, store_type)
            assert numpy.allclose(af[n], a)


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
@pytest.mark.parametrize("with_update", [True, False])
def test_modify(tmp_path, with_update, copy_arrays, lazy_load):
    # make a file
    store = DirectoryStore(tmp_path / "zarr_array")
    arr = create_zarray(store=store)
    tree = {"arr": arr}
    fn = tmp_path / "test.asdf"
    af = asdf.AsdfFile(tree)
    af.write_to(fn)

    # open the file, modify the array
    with asdf.open(fn, mode="rw", copy_arrays=copy_arrays, lazy_load=lazy_load) as af:
        assert af["arr"][0, 0] != 42
        af["arr"][0, 0] = 42
        # now modify
        assert af["arr"][0, 0] == 42
        # also check the original array
        assert arr[0, 0] == 42
        if with_update:
            af.update()

    # reopen the file, check for the modification
    with asdf.open(fn, mode="rw", copy_arrays=copy_arrays, lazy_load=lazy_load) as af:
        assert af["arr"][0, 0] == 42


class CustomStore(zarr.storage.Store, UserDict):
    # an 'unknown' custom storage class
    pass


@pytest.mark.skip("ASDF Converters aren't aware of the open mode")
@pytest.mark.parametrize("mode", ["r", "rw"])
def test_open_mode(tmp_path, mode):
    store = DirectoryStore(tmp_path / "zarr_array")
    arr = create_zarray(store=store)
    tree = {"arr": arr}
    fn = tmp_path / "test.asdf"
    af = asdf.AsdfFile(tree)
    af.write_to(fn)

    with asdf.open(fn, mode=mode) as af:
        if mode == "r":
            # array changes during an open 'r' should fail
            with pytest.raises():
                af["arr"][0, 0] = 1
        elif mode == "rw":
            # array changes should be allowed during 'rw'
            af["arr"][0, 0] = 1
        else:
            raise Exception(f"Unknown mode {mode}")


@pytest.mark.parametrize("meta_store", [True, False])
def test_to_internal(meta_store):
    if meta_store:
        zarr = create_zarray(store=KVStore({}), chunk_store=TempStore())
    else:
        zarr = create_zarray(store=TempStore())
    internal = asdf_zarr.storage.to_internal(zarr)
    assert isinstance(internal.chunk_store, asdf_zarr.storage.InternalStore)
    # the store shouldn't be wrapped if it's not used for chunks
    if zarr.store is not zarr.chunk_store:
        assert isinstance(internal.store, KVStore)
    # calling it a second time shouldn't re-wrap the store
    same = asdf_zarr.storage.to_internal(internal)
    assert same.chunk_store is internal.chunk_store

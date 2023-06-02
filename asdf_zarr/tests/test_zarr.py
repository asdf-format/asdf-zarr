from collections import UserDict
import itertools

import asdf
import numpy
import pytest
import zarr
from zarr.storage import DirectoryStore, NestedDirectoryStore


def create_zarray(shape=None, chunks=None, dtype="f8", store=None):
    if shape is None:
        shape = (6, 9)
    if chunks is None:
        chunks = [max(1, d // 3) for d in shape]
    arr = zarr.creation.create((6, 9), store=store, chunks=chunks, dtype=dtype, compressor=None)
    for chunk_index in itertools.product(*[range(c) for c in arr.cdata_shape]):
        inds = []
        for i, c in zip(chunk_index, arr.chunks):
            inds.append(slice(i * c, (i + 1) * c))
        arr[tuple(inds)] = i
    return arr


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
@pytest.mark.parametrize("compression", ["input", "zlib"])
@pytest.mark.parametrize("store_type", [DirectoryStore, NestedDirectoryStore])
def test_write_to(tmp_path, copy_arrays, lazy_load, compression, store_type):
    store1 = store_type(tmp_path / "zarr_array_1")
    store2 = store_type(tmp_path / "zarr_array_2")

    arr1 = create_zarray(store=store1)
    arr2 = create_zarray(store=store2)
    arr2[:] = arr2[:] * -2
    tree = {"arr1": arr1, "arr2": arr2}

    fn = tmp_path / "test.asdf"
    af = asdf.AsdfFile(tree)
    af.write_to(fn, all_array_compression=compression)

    with asdf.open(fn, mode="r", copy_arrays=copy_arrays, lazy_load=lazy_load) as af:
        for n, a in (("arr1", arr1), ("arr2", arr2)):
            assert isinstance(af[n], zarr.core.Array)
            # for these tests, data should not be converted to a different storage format
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

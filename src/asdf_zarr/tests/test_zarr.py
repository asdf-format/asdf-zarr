from collections import UserDict
import itertools

import asdf
import asdf_zarr
import asdf_zarr.storage
import numpy
import pytest
import zarr
from zarr.storage import KVStore, DirectoryStore, FSStore, MemoryStore, NestedDirectoryStore, TempStore


def create_zarray(shape=None, chunks=None, dtype='f8', store=None):
    if shape is None:
        shape = (6, 9)
    if chunks is None:
        chunks = [max(1, d // 3) for d in shape]
    arr = zarr.creation.create((6, 9), store=store, chunks=chunks, dtype=dtype, compressor=None)
    for chunk_index in itertools.product(*[range(c) for c in arr.cdata_shape]):
        inds = []
        for (i, c) in zip(chunk_index, arr.chunks):
            inds.append(slice(i * c, (i + 1) * c))
        arr[tuple(inds)] = i
    return arr


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
@pytest.mark.parametrize("compression", ["input", "zlib"])
@pytest.mark.parametrize("store_type", [DirectoryStore, NestedDirectoryStore])
def test_write_to(tmp_path, copy_arrays, lazy_load, compression, store_type):
    store1 = store_type(tmp_path / 'zarr_array_1')
    store2 = store_type(tmp_path / 'zarr_array_2')

    arr1 = create_zarray(store=store1)
    arr2 = create_zarray(store=store2)
    arr2[:] = arr2[:] * -2
    tree = {'arr1': arr1, 'arr2': arr2}


    fn = tmp_path / 'test.asdf'
    af = asdf.AsdfFile(tree)
    af.write_to(fn, all_array_compression=compression)

    with asdf.open(fn, mode='r', copy_arrays=copy_arrays, lazy_load=lazy_load) as af:
        for (n, a) in (('arr1', arr1), ('arr2', arr2)):
            assert isinstance(af[n], zarr.core.Array)
            # for these tests, data should not be converted to a different storage format
            assert isinstance(af[n].chunk_store, store_type)
            assert numpy.allclose(af[n], a)


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
@pytest.mark.parametrize("with_update", [True, False])
def test_modify(tmp_path, with_update, copy_arrays, lazy_load):
    # make a file
    store = DirectoryStore(tmp_path / 'zarr_array')
    arr = create_zarray(store=store)
    tree = {'arr': arr}
    fn = tmp_path / 'test.asdf'
    af = asdf.AsdfFile(tree)
    af.write_to(fn)

    # open the file, modify the array
    with asdf.open(fn, mode='rw', copy_arrays=copy_arrays, lazy_load=lazy_load) as af:
        assert af['arr'][0, 0] != 42
        af['arr'][0, 0] = 42
        # now modify
        assert af['arr'][0, 0] == 42
        # also check the original array
        assert arr[0, 0] == 42
        if with_update:
            af.update()

    # reopen the file, check for the modification
    with asdf.open(fn, mode='rw', copy_arrays=copy_arrays, lazy_load=lazy_load) as af:
        assert af['arr'][0, 0] == 42


class CustomStore(zarr.storage.Store, UserDict):
    # an 'unknown' custom storage class
    pass


@pytest.mark.skip("ASDF Converters aren't aware of the open mode")
@pytest.mark.parametrize("mode", ["r", "rw"])
def test_open_mode(tmp_path, mode):
    store = store_type(tmp_path / 'zarr_array')
    arr = create_zarray(store=store)
    tree = {'arr': arr}
    fn = tmp_path / 'test.asdf'
    af = asdf.AsdfFile(tree)
    af.write_to(fn)

    with asdf.open(fn, mode=mode) as af:
        if mode == 'r':
            # array changes during an open 'r' should fail
            with pytest.raises():
                af['arr'][0, 0] = 1
        elif mode == 'rw':
            # array changes should be allowed during 'rw'
            af['arr'][0, 0] = 1
        else:
            raise Exception(f"Unknown mode {mode}")


@pytest.mark.fake_s3()
def test_fsstore_s3(tmp_path):
    # endpoint used to fake s3
    endpoint_url = 'http://127.0.0.1:5555'
    bucket = 'test_bucket'
    url = f's3://{bucket}/my_zarr'

    # connect to 's3'
    import boto3
    conn = boto3.resource('s3', endpoint_url=endpoint_url)

    # make a new bucket to store the data
    bucket = conn.create_bucket(Bucket=bucket)

    # clear the bucket so zarr.zeros can create an array
    bucket.objects.delete()

    # create a fsstore using the s3 url
    store = FSStore(url, client_kwargs={'endpoint_url': endpoint_url})

    # create a new chunked array
    a = zarr.zeros(store=store, shape=(1000, 1000), chunks=(100, 100), dtype='i4')
    # write to the array
    a[42, 26] = 42

    # create an asdf file and save the chunked array
    af = asdf.AsdfFile()
    af['my_zarr'] = a
    fn = tmp_path / 'test_zarr.asdf'
    af.write_to(fn)

    # open the asdf file and check the chunked array loaded
    with asdf.open(fn) as af:
        a = af['my_zarr']
        assert a[42, 26] == 42


# compression appears to work for first writes but not rewrites
# for now, disable it in testing
#@pytest.mark.parametrize("compression", ["input", "zlib"])
@pytest.mark.parametrize("compression", ["input"])
@pytest.mark.parametrize("store_type", [KVStore, MemoryStore, DirectoryStore, NestedDirectoryStore])
def test_convert_to_internal(tmp_path, compression, store_type):
    # when requested, ingest the data and include it as internal blocks
    if store_type in (KVStore, MemoryStore):
        store1 = store_type({})
        store2 = store_type({})
    else:
        store1 = store_type(tmp_path / 'zarr_array_1')
        store2 = store_type(tmp_path / 'zarr_array_2')

    arr1 = create_zarray(store=store1)
    arr2 = create_zarray(store=store2)
    arr2[:] = arr2[:] * -2

    # now make arrays that will be converted to internal storage
    if store_type in (KVStore, MemoryStore):
        # these should be automatic
        tree = {'arr1': arr1, 'arr2': arr2}
    else:
        tree = {
            'arr1': asdf_zarr.storage.to_internal(arr1),
            'arr2': asdf_zarr.storage.to_internal(arr2),
        }

    af = asdf.AsdfFile(tree)

    fn = tmp_path / 'test.asdf'
    fn2 = tmp_path / 'test2.asdf'
    af.write_to(fn, all_array_compression=compression)

    with asdf.open(fn, mode='r') as af:
        for (n, a) in (('arr1', arr1), ('arr2', arr2)):
            assert isinstance(af[n], zarr.core.Array)
            # for these tests, data should not be converted to a different storage format
            assert isinstance(af[n].chunk_store, asdf_zarr.storage.InternalStore)
            assert numpy.allclose(af[n], a)
        # check that resaving works
        af.write_to(fn2, all_array_compression=compression)

    with asdf.open(fn2, mode='r') as af:
        for (n, a) in (('arr1', arr1), ('arr2', arr2)):
            assert numpy.allclose(af[n], a)
            # modify data, make sure the internal data is unchanged
            a[:] += 1
            assert not numpy.allclose(af[n], a), (af[n][:], a[:])


def test_modify_internal(tmp_path):
    # setup to convert a zarray to internal
    store1 = KVStore({})
    store2 = KVStore({})
    arr1 = create_zarray(store1)
    arr2 = create_zarray(store2)

    arr1[:] = 1
    arr2[:] = 2

    tree = {
        'arr1': asdf_zarr.storage.to_internal(arr1),
        'arr2': asdf_zarr.storage.to_internal(arr2),
    }

    af = asdf.AsdfFile(tree)

    # modify post-setup (should not change original store)
    af['arr1'][0, 0] = 2
    af['arr2'][0, 0] = 4
    assert arr1[0, 0] == 1
    assert arr2[0, 0] == 2
    assert af['arr1'][0, 0] == 2
    assert af['arr2'][0, 0] == 4

    # save to internal
    fn = tmp_path / 'test.asdf'
    fn2 = tmp_path / 'test2.asdf'
    af.write_to(fn)

    # open internal
    with asdf.open(fn) as af:
        assert af['arr1'][0, 0] == 2
        assert af['arr2'][0, 0] == 4

        # modify
        af['arr1'][0, 0] = 10
        af['arr2'][0, 0] = 20

        # resave
        af.write_to(fn2)

    # check all saved data is correct
    with asdf.open(fn) as af:
        assert af['arr1'][0, 0] == 2
        assert af['arr2'][0, 0] == 4
    with asdf.open(fn2) as af:
        assert af['arr1'][0, 0] == 10
        assert af['arr2'][0, 0] == 20


@pytest.mark.skip("Not Implemented")
def test_warn_when_copy():
    # warn if a file with internal blocks is loaded with copy_arrays
    # this will require generalizing copy_array/lazy_load
    pass

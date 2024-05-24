import importlib

# TODO: v3 is the default branch yet the version still reports as 2.x
# for now, look for the "v2" submodule
_is_v3 = importlib.util.find_spec("zarr.v2") is not None

if _is_v3:
    import zarr.v2 as zarr
    import zarr.v2.storage as storage
else:
    import zarr
    from zarr import storage

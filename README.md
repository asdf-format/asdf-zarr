This is an experimental asdf extension to allow writing and reading
of externally stored zarr arrays.

Some optional tests (ones that access s3) expect a fake s3 running at
127.0.0.1:5555 (see [examples/s3_zarr_storage/run_server.sh]),
a 'fake_s3' pytest mark (`pytest -m fake_s3`), and the boto3 package.

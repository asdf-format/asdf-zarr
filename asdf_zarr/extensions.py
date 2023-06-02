import asdf

from .converter import ZarrConverter


class ZarrExtension(asdf.extension.Extension):
    extension_uri = "asdf://stsci.edu/example-project/tags/zarr-1.0.0"
    tags = ["asdf://stsci.edu/example-project/tags/zarr-1.0.0"]
    converters = [ZarrConverter()]


def get_extensions():
    return [ZarrExtension()]

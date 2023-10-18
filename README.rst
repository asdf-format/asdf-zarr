ASDF serialization support for Zarr
-----------------------------------

.. image:: https://github.com/asdf-format/asdf-zarr/workflows/CI/badge.svg
    :target: https://github.com/asdf-format/asdf-zarr/actions
    :alt: CI Status
.. image:: https://codecov.io/gh/asdf-format/asdf-zarr/branch/main/graphs/badge.svg
    :target: https://codecov.io/gh/asdf-format/asdf-zarr
    :alt: Codecov

This packages includes an extension for the Python library
`asdf <https://asdf.readthedocs.io/en/latest/>`__ to add support
for reading and writing chunked
`Zarr <https://zarr.readthedocs.io/en/stable/>`__ arrays.


Installation
------------

This extension is available on PyPI. We are actively developing
this extension and until a stable release (`1.0.0`) is made it
is possible that breaking changes will be introduced. If you
are using this extension please let us know so we can look into
adding your project to downstream testing in our CI.

.. code-block:: console

    $ pip install asdf-zarr

Alternatively this extension can be installed by cloning
and installing the git repository.

.. code-block:: console

    $ git clone https://github.com/asdf-format/asdf-zarr
    $ cd asdf-zarr
    $ pip install .


Usage
-----

For background on ASDF array storage and examples
of how to use this extension see the notebooks
subdirectory of this repository.


Testing
-------

`pytest <https://docs.pytest.org>`__ is used for testing.
Tests can be run (from the source checkout of this repository):

.. code-block:: console

    $ pytest


Contributing
------------

We welcome feedback and contributions to this project.

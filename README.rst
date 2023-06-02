ASDF serialization support for Zarr
-----------------------------------

.. image:: https://github.com/asdf-format/asdf-zarr/workflows/CI/badge.svg
    :target: https://github.com/asdf-format/asdf-zarr/actions
    :alt: CI Status

This packages includes a plugin for the Python library
`asdf <https://asdf.readthedocs.io/en/latest/>`__ to add support
for reading and writing chunked
`Zarr <https://zarr.readthedocs.io/en/stable/>`__ arrays.


Installation
------------

This plugin is not yet stable and released on
PyPi and requires features only available in the
current development head of ASDF.

.. code-block:: console

    $ pip install git+https://github.com/asdf-format/asdf
    $ pip install git+https://github.com/asdf-format/asdf-zarr

Or alternatively by cloning and installing each package.

.. code-block:: console

    $ git clone https://github.com/asdf-format/asdf
    $ cd asdf
    $ pip install .
    $ cd ../
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

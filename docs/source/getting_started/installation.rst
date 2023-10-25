.. _installation_ref:

Installation
============
Wake-T is tested on Python 3.8 and above, so make sure to have an up-to-date version
in order to avoid potential issues. It is recommended to install Wake-T in its
own virtual environment, either using ``venv`` or ``conda``. The installation should be
carried out with ``pip``, since Wake-T is available on `PyPI <https://pypi.org/project/Wake-T/>`_
and `GitHub <https://github.com/AngelFP/Wake-T.git>`_, but not on any Anaconda repository.


Dependencies
------------
Wake-T relies on the following packages:

* `NumPy <https://pypi.org/project/numpy/>`_ - NumPy arrays and operations are at the core of Wake-T.
* `Numba <https://pypi.org/project/numba/>`_ - Just-in-time compilation of compute-intensive methods.
* `SciPy <https://pypi.org/project/scipy/>`_ - Physical constants and numerical tools.
* `APtools <https://pypi.org/project/APtools/>`_ - Various utilities for beam diagnostics, importing/saving particle data from/to other codes (e.g. ASTRA, CSRtrack), generation of particle distributions and other tools.
* `openPMD-api <https://pypi.org/project/openPMD-api/>`_ - I/O of simulation data.

They are all automatically installed together with Wake-T.

PyPI (stable)
-------------
.. image:: https://img.shields.io/pypi/v/Wake-T.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/Wake-T/

Install the latest stable release of Wake-T from `PyPI <https://pypi.org/project/Wake-T/>`_
using ``pip``::

    pip install Wake-T

    
Latest (development) version from GitHub
----------------------------------------
.. image:: https://github.com/AngelFP/Wake-T/actions/workflows/test-package.yml/badge.svg
   :target: https://github.com/AngelFP/Wake-T/actions

The latest release on PyPI might not be fully up to date with the newest additions
to the code. If you want to stay at the bleeding edge of new features and bug fixes
(and perhaps new bugs too), install Wake-T directly from GitHub using ``pip`` and ``git``:

.. code::

   pip install -U git+https://github.com/AngelFP/Wake-T.git

The ``-U`` flag will make sure the package is upgraded to the newest version, if you already had Wake-T installed.

Alternatively, if you prefer cloning the repository with git, you can install it by doing

.. code::

   git clone https://github.com/AngelFP/Wake-T.git
   cd Wake-T
   pip install .


Optional dependencies
---------------------
If you are running Wake-T simulations, you will most likely generate output data using the openPMD diagnostics.
In order to visualize and analyze this data, the following packages are recommended:

* `openPMD-viewer <https://github.com/openPMD/openPMD-viewer>`_ - Data analysis and quick visualization in Jupyter notebooks.
* `VisualPIC <https://github.com/AngelFP/VisualPIC>`_ - Data analysis and 3D visualization.


Installation on particular HPC clusters
---------------------------------------
Dedicated instructions for installing Wake-T in particular HPC clusters can be found in the following links:

.. toctree::
   :maxdepth: 1

   installation_maxwell
   installation_juwels
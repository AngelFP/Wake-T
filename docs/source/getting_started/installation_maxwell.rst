Installation on Maxwell (DESY)
==============================

This page describes the steps for installing Wake-T on the DESY `Maxwell <https://confluence.desy.de/display/MXW/Maxwell+Cluster/>`_ Cluster.


Creating the Python environment
-------------------------------

The first step will be to create a virtual environment for Wake-T. This is in principle optional,
but a good practice which will keep Wake-T and its dependencies isolated from your other Python projects.

Using the provided Anaconda
```````````````````````````

The easiest way of creating a virtual environment is using the Anaconda
installation provided by IT:

.. code::

    module load anaconda
    conda create -n wake_t_env python=3.8
    source activate wake_t_env

Installing your own Miniconda
`````````````````````````````

If you prefer more control over your Python environments, you can also install your own
Miniconda by typing

.. code::

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

.. note::

    Install Miniconda and not a full Anaconda distribution, as this would most likely fill up your available quota.

(Optional) After completing the installation, use

.. code::

    conda config --set auto_activate_base false

to prevent the conda base environment from being automatically activated every time you log into Maxwell.

Now you can create and activate the environment

.. code::

    conda create -n wake_t_env python=3.8
    conda activate wake_t_env
    
.. note::

    If you plan on using Wake-T from the Maxwell JupyterHub, you need to manually register this environment
    as a Jupyter Notebook kernel:

    .. code::

        conda install ipykernel
        python -m ipykernel install --user --name wake_t_env --display-name "Python (wake_t_env)"

    This should not be necessary if you are using Maxwell's Anaconda.


Installing Wake-T
-----------------

To get access to the latest features, install directly from GitHub:

.. code::

    pip install -U git+https://github.com/AngelFP/Wake-T.git


Installing openPMD-viewer and VisualPIC
---------------------------------------

To analyze and visualize the data, get the latest versions of the openPMD-viewer and VisualPIC:

.. code::

    pip install -U git+https://github.com/openPMD/openPMD-viewer.git@dev
    pip install -U git+https://github.com/AngelFP/VisualPIC.git@general_redesign

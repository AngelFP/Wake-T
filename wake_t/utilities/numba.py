"""This module contains custom definitions of the numba decorators."""

import os
from numba import njit, __version__ as numba_version


if numba_version == '0.57.0':
    raise RuntimeError(
        'Wake-T is incompatible with numba 0.57.0.\n'
        'Please install either a later or an earlier version.'
    )


# Check if the environment variable WAKET_DISABLE_CACHING is set to 1
# and in that case, disable caching
caching = True
if 'WAKET_DISABLE_CACHING' in os.environ:
    if int(os.environ['WAKET_DISABLE_CACHING']) == 1:
        caching = False


# Define custom njit decorator for serial methods.
def njit_serial(*args, **kwargs):
    return njit(*args, cache=caching, **kwargs)


# Define custom njit decorator for parallel methods.
def njit_parallel(*args, **kwargs):
    return njit(*args, cache=caching, parallel=True, **kwargs)

import os
from numba import njit

# Check if the environment variable WAKET_DISABLE_CACHING is set to 1
# and in that case, disable caching
caching = True
if 'WAKET_DISABLE_CACHING' in os.environ:
    if int(os.environ['WAKET_DISABLE_CACHING']) == 1:
        caching = False

# Set the function njit
njit_func = njit(cache=caching)

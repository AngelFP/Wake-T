"""Defines a progress bar to be used by the Tracker."""

import sys
import warnings

from tqdm import tqdm


# Avoid showing clamping warnings from the progress bar.
warnings.filterwarnings('ignore', '.*clamping.*', )


def get_progress_bar(description, total_length, disable):
    """Get progress bar for the tracker.

    Parameters
    ----------
    description : str
        Description to be appended to start of the progress bar.
    total_length : float
        Total length in metres of the stage to be tracked.
    disable : bool
        Whether to disable (not show) the progress bar.

    Returns
    -------
    A tqdm progress bar.
    """
    l_bar = "{desc}: {percentage:3.0f}%|"
    r_bar = "| {n:.6f}/{total:.6f} {unit} [{elapsed}]"
    progress_bar = tqdm(
        desc=description,
        total=total_length,
        unit='m',
        bar_format=l_bar + "{bar}" + r_bar,
        file=sys.stdout,
        disable=disable
    )
    return progress_bar

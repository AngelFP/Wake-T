"""Defines a progress bar to be used by the Tracker."""

from tqdm import tqdm
import sys

def get_progress_bar(description, total_length):
    """Get progress bar for the tracker.

    Parameters
    ----------
    description : str
        Description to be appended to start of the progress bar.
    total_length : float
        Total length in metres of the stage to be tracked.

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
        dynamic_ncols=True
    )
    return progress_bar

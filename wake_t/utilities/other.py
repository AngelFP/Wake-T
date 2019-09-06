""" Contains other utilities """

import sys


def print_progress_bar(pre_string, step, total_steps):
    n_dash = int(round(step/total_steps*20))
    n_space = 20 - n_dash
    status = pre_string + '[' + '-'*n_dash + ' '*n_space + '] '
    if step < total_steps:
        status += '\r'
    sys.stdout.write(status)
    sys.stdout.flush()

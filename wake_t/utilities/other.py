"""Contains other utilities"""

import sys
import time

import numpy as np


def print_progress_bar(pre_string, step, total_steps):
    n_dash = int(round(step / total_steps * 20))
    n_space = 20 - n_dash
    status = pre_string + "[" + "-" * n_dash + " " * n_space + "] "
    if step < total_steps:
        status += "\r"
    sys.stdout.write(status)
    sys.stdout.flush()


def radial_gradient(fld, dr):
    """
    Calculate the radial gradient of a 2D r-z field.

    To obtain an accurate derivative on axis, a wider array which contains
    the initial field and its mirrored view along the axis is created. The
    gradient of this array is computed and only its upper half is returned.

    Parameters
    ----------
    fld : ndarray
        A 2D array containing the original r-z field.
    dr : float
        Radial separation between grid points.

    """
    n_r = fld.shape[1]
    fld_with_mirror = np.concatenate((fld[:, ::-1], fld), axis=1)
    return np.gradient(fld_with_mirror, dr, axis=1)[:, n_r:]


class ProfileRange:
    def __init__(self, name):
        self.name = name
        self.time_begin = time.time()
        self.time_counter = 0.0
        self.num_calls = 0
        self.first_time = 0.0

    def start(self):
        self.time_begin = time.time()

    def stop(self):
        delta = time.time() - self.time_begin
        if self.num_calls == 0:
            self.first_time = delta
        self.time_counter += delta
        self.num_calls += 1

    def get_results(self, total_time):
        return (
            self.name,
            f"{self.num_calls}",
            f"{1000 * self.time_counter / max(self.num_calls, 1):.04g} ms",
            f"{1000 * self.first_time:.04g} ms",
            f"{self.time_counter:.04g} s",
            f"{self.time_counter / total_time:.02%}",
        )


class Profiler(dict):
    def __init__(self):
        self.enable = False

    def __missing__(self, name):
        self[name] = ProfileRange(name)
        return self[name]

    def __enter__(self):
        self.clear()
        self.enable = True
        self.total_begin = time.time()

    def __exit__(self, *args):
        self.print_results()
        self.clear()
        self.enable = False

    def start(self, name):
        if self.enable:
            self[name].start()

    def stop(self, name):
        if self.enable:
            self[name].stop()

    def print_results(self):
        if self.enable:
            total_time = time.time() - self.total_begin
            print(f"\nTotal time: {total_time:.04g} s\n")
            if len(self) > 0:
                table = [p.get_results(total_time) for p in self.values()]
                table.sort(key=lambda x: float(x[-2][:-2]), reverse=True)
                table.insert(0, ("Name", "NCalls", "Avg", "First", "Total", "%"))

                num_chars = np.zeros(len(table[0]), dtype=np.int64)

                for row in table:
                    for i in range(len(row)):
                        num_chars[i] = max(num_chars[i], len(row[i]))

                num_chars += 2

                print(np.sum(num_chars) * "-")
                for j in range(len(table)):
                    for i in range(len(table[j])):
                        print(
                            f"{table[j][i]:{'<' if i == 0 else '>'}{num_chars[i]}}",
                            end="",
                        )
                    print()
                    if j == 0:
                        print(np.sum(num_chars) * "-")
                print(np.sum(num_chars) * "-")


_global_profiler = Profiler()


def Profiling():
    return _global_profiler


def ProfStart(name):
    _global_profiler.start(name)


def ProfStop(name):
    _global_profiler.stop(name)

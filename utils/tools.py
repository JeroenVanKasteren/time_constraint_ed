"""
Static functions for the project.
"""

import numpy as np
import pandas as pd
from time import strptime


def def_sizes(dim):
    """Docstring."""
    sizes = np.zeros(len(dim), np.int32)
    sizes[-1] = 1
    for i in range(len(dim) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * dim[i + 1]
    return sizes


def strip_split(x):
    return np.array([float(i) for i in x.strip('[]').split(', ')])


def get_time(time_string):
    """Read in time in formats (D)D-HH:MM:SS, (H)H:MM:SS, or (M)M:SS."""
    if ((time_string is not None) & (not pd.isnull(time_string)) &
            (time_string != np.inf)):
        if '-' in time_string:
            days, time = time_string.split('-')
        elif time_string.count(':') == 1:
            days, time = 0, '0:'+time_string
        else:
            days, time = 0, time_string
        x = strptime(time, '%H:%M:%S')
        return (((int(days) * 24 + x.tm_hour) * 60 + x.tm_min) * 60
                + x.tm_sec - 60)
    else:
        return np.Inf


def sec_to_time(time):
    """Convert seconds to minutes and return readable format."""
    time = int(time)
    if time >= 60*60:
        return f"Time (HH:MM:SS): {time // (60*60):02d}:{(time // 60) % 60:02d}:{time % 60:02d}"
    else:
        return f"Time (MM:SS): {time // 60:02d}:{time % 60:02d}"


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
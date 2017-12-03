"""This module implements a class that..."""
from __future__ import print_function, unicode_literals

import os

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class SheetManagerDBError(OSError):
    pass


def path2name(path):
    path, name = os.path.split(path)
    while not name:
        path, name = os.path.split(path)
        if len(path) <= 1:
            return None
    return name

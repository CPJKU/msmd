"""This module implements a class that..."""
from __future__ import print_function

import logging
import os
import yaml

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class MSMDMetadataMixin(object):
    """Implements metadata load/dump mechanism.

    However, without piece folder name, it needs to be supplied
    explicitly."""
    META_FNAME = 'meta.yml'

    def __init__(self):
        self.metadata = {}

    def _load_metadata_from(self, folder):
        metafile = os.path.join(folder, self.META_FNAME)
        if not os.path.isfile(metafile):
            logging.info('Metadata file is not available!'
                         ' Returning empty dict.')
            return dict()

        with open(metafile, 'r') as hdl:
            metadata = yaml.load_all(hdl)

        return metadata

    def _dump_metadata_to(self, folder):
        metafile = os.path.join(folder, self.META_FNAME)
        if not self.metadata:
            return
        with open(metafile, 'w') as hdl:
            yaml.dump(self.metadata, hdl)

    def load_metadata(self):
        if not hasattr(self, 'metadata_folder'):
            raise SheetManagerDBError('MSMDMetadataMixin can only operate'
                                      ' on classes that have a metadata_folder'
                                      ' attribute!')
        return self._load_metadata_from(getattr(self, 'metadata_folder'))

    def dump_metadata(self):
        if not hasattr(self, 'metadata_folder'):
            raise SheetManagerDBError('MSMDMetadataMixin can only operate'
                                      ' on classes that have a metadata_folder'
                                      ' attribute!')
        self._dump_metadata_to(getattr(self, 'metadata_folder'))


class SheetManagerDBError(OSError):
    pass


def path2name(path):
    path, name = os.path.split(path)
    while not name:
        path, name = os.path.split(path)
        if len(path) <= 1:
            return None
    return name

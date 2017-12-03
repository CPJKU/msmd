"""This module implements a class that represents a score of a piece."""
from __future__ import print_function

import logging
import os

import yaml

from sheet_manager.data_model.util import SheetManagerDBError, path2name

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class Score(object):
    """The Score class represents one score of a piece. Each score has a PDF file
    as an authority, and then several views:

    * Images -- one ``*.png`` file per page
    * Coords -- the bounding boxes of systems, upper and lower points of bars,
                and centroids of notes.

    In the near future, there will also be:

    * MuNG (MUSCIMA++ Notation Graph)

    """
    DEFAULT_META_FNAME = 'meta.yml'

    def __init__(self, folder, piece_name):
        """Initialize the Score.

        :param folder: The directory in which the score should be initialized.
            The name of the Score is derived from the name of this directory.

        :param piece_name: Name of the Piece to which the Score belongs.
        """
        if not os.path.isdir(folder):
            raise SheetManagerDBError('Performance initialized with'
                                      ' non-existent directory: {0}'
                                      ''.format(folder))
        self.folder = folder
        name = path2name(folder)
        self.name = name
        self.piece_name = piece_name

        self.pdf_file = self.discover_pdf()

        self.metadata = self.load_metadata()

        # Shortcut to standard views: img and coords
        self.img_dir = os.path.join(self.folder, 'img')
        self.coords_dir = os.path.join(self.folder, 'coords')
        self._ensure_directory_structure()

        self.views = self.collect_views()

    @property
    def n_pages(self):
        """Derived from image view."""
        n_pages = len([f for f in os.listdir(self.img_dir)
                       if not f.startswith('.')])
        if n_pages == 0:
            logging.warning('Counting pages in score {0}: it seems'
                            ' that there are no images generated yet!'
                            ' Returning 0, but do not trust this number.'
                            ''.format(self.folder))
        return n_pages

    @property
    def image_files(self):
        return [os.path.join(self.img_dir, img)
                for img in os.listdir(self.img_dir)
                if not img.startswith('.')]

    def discover_pdf(self):
        available_pdf_files = [f for f in os.listdir(self.folder)
                               if f.endswith('.pdf')]
        if len(available_pdf_files) == 0:
            raise SheetManagerDBError('Instantiated a Score without the PDF'
                                      ' authority document: {0}'.format(self.folder))
        if len(available_pdf_files) > 1:
            raise SheetManagerDBError('Instantiated a Score with more than one PDF'
                                      ' authority document: {0}'.format(self.folder))
        pdf_fname = available_pdf_files[0]
        pdf_file = os.path.join(self.folder, pdf_fname)
        return pdf_file

    def clear_images(self):
        """Clears all of the score images."""
        for f in os.listdir(self.img_dir):
            os.unlink(os.path.join(self.img_dir, f))

    def collect_views(self):
        """Returns all available score views."""
        return {v: os.path.join(self.folder, v)
                for v in os.listdir(self.folder)
                if os.path.isdir(os.path.join(self.folder, v))}

    def load_metadata(self):
        """Loads arbitrary YAML descriptors with the default name (meta.yml)."""
        metafile = os.path.join(self.folder, self.DEFAULT_META_FNAME)
        if not os.path.isfile(metafile):
            logging.warn('performance {0} has no metadata file: {1}'
                         ''.format(self.name, self.DEFAULT_META_FNAME))
            return dict()

        with open(metafile, 'r') as hdl:
            metadata = yaml.load_all(hdl)

        return metadata

    def _ensure_directory_structure(self):
        if not os.path.isdir(self.img_dir):
            os.mkdir(self.img_dir)
        if not os.path.isdir(self.coords_dir):
            os.mkdir(self.coords_dir)


"""This module implements a class that represents a score of a piece."""
from __future__ import print_function

import logging
import os
import shutil
import time
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
            raise SheetManagerDBError('Score initialized with'
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

    def update(self):
        self.pdf_file = self.discover_pdf()
        self._ensure_directory_structure()
        self.metadata = self.load_metadata()
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

    @staticmethod
    def format_page_name(page):
        """Implements the naming convention for page numbers in the view
        files.

        If ``page`` is an integer, the method adds 1 and formats it
        as a two-digit string, with a leading zero if necessary.

        If ``page`` is anything else than an integer, applies ``str()``.
        """
        if isinstance(page, int):
            page_str = '{0:02d}'.format(page + 1)
        else:
            page_str = str(page)
        return page_str

    def add_paged_view(self, view_name, view_data_per_page,
                       file_fmt,
                       binary=False,
                       prefix=None,
                       overwrite=False):
        """Adds a view of the Score from the given data. The data is expected
        to be a dict with page numbers as keys. The values are expected to be
        already formatted so that they can be simply dumped into an open file
        stream.

        Filenames in the view will be constructed as ``prefix_page.file_fmt``
        from the arguments ``file_fmt`` (required), ``prefix`` (not required),
        and ``page`` is derived from the ``view_data_per_page`` keys (if the
        keys are strings, then they are taken as-is; if the page keys are
        integers, they are converted to a 2-digit string).

        If ``overwrite`` is set and a view with ``view_name`` already exists,
        it will be cleared and overwritten.
        """
        self.update()
        if view_name in self.views:
            if overwrite:
                logging.warning('Score {0}: view {1} already exists;'
                                ' overwriting...'.format(self.name,
                                                         view_name))
                time.sleep(3)
                self.clear_view(view_name)
            else:
                logging.warning('Score {0}: view {1} already exists;'
                                ' will not do anything.'.format(self.name,
                                                                view_name))
                return

        if file_fmt.startswith('.'):
            file_fmt = file_fmt[1:]

        view_path = os.path.join(self.folder, view_name)
        os.mkdir(view_path)
        for page in view_data_per_page:

            page_str = self.format_page_name(page)
            page_fname = '{0}.{1}'.format(page_str, file_fmt)
            if prefix is not None:
                page_fname = '{0}_'.format(prefix) + page_fname
            page_file_path = os.path.join(view_path, page_fname)

            data = view_data_per_page[page]

            mode = 'w'
            if binary:
                mode = 'wb'

            with open(page_file_path, mode=mode) as hdl:
                hdl.write(data)

        self.update()

    def view_files(self, view_name):
        """Return a list of the paths to all (non-hidden) files
        in the view."""
        self.update()
        if view_name not in self.views:
            raise SheetManagerDBError('Score {0}: requested view {1}'
                                      ' not available!'
                                      ''.format(self.name, view_name))
        view_dir = self.views[view_name]
        return [os.path.join(view_dir, f) for f in os.listdir(view_dir)
                if (not f.startswith('.')) and
                   (os.path.isfile(os.path.join(view_dir, f)))]

    def clear_view(self, view_name):
        """Removes the given view."""
        self.update()
        if view_name not in self.views:
            raise SheetManagerDBError('Score {0}: requested clearing view'
                                      ' {1}, but this view does not exist!'
                                      ' (Available views: {2})'
                                      ''.format(self.name, view_name,
                                                self.views.keys()))
        shutil.rmtree(self.views[view_name])
        self.update()

    def collect_views(self):
        """Returns all available score views."""
        return {v: os.path.join(self.folder, v)
                for v in os.listdir(self.folder)
                if os.path.isdir(os.path.join(self.folder, v))}

    def load_metadata(self):
        """Loads arbitrary YAML descriptors with the default name (meta.yml).
        """
        metafile = os.path.join(self.folder, self.DEFAULT_META_FNAME)
        if not os.path.isfile(metafile):
            logging.warn('Score {0} has no metadata file: {1}'
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


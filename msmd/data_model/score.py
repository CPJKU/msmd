"""This module implements a class that represents a score of a piece."""
from __future__ import print_function

import collections
import logging
import os
import pprint
import shutil
import time

import cv2
import yaml
from muscima.graph import NotationGraph
from muscima.inference_engine_constants import InferenceEngineConstants
from muscima.io import parse_cropobject_list

from msmd.data_model.util import SheetManagerDBError, path2name, MSMDMetadataMixin

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class Score(MSMDMetadataMixin):
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
        super(Score, self).__init__()

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
    def metadata_folder(self):
        return self.folder

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
                for img in sorted(os.listdir(self.img_dir))
                if not img.startswith('.')]

    def load_images(self):
        images = [cv2.imread(f, 0) for f in self.image_files]
        return images

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
                logging.info('Score {0}: view {1} already exists;'
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
        return [os.path.join(view_dir, f) for f in sorted(os.listdir(view_dir))
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

    # def load_metadata(self):
    #     """Loads arbitrary YAML descriptors with the default name (meta.yml).
    #     """
    #     metafile = os.path.join(self.folder, self.DEFAULT_META_FNAME)
    #     if not os.path.isfile(metafile):
    #         logging.info('Score {0} has no metadata file: {1}'
    #                      ''.format(self.name, self.DEFAULT_META_FNAME))
    #         return dict()
    #
    #     with open(metafile, 'r') as hdl:
    #         metadata = yaml.load_all(hdl)
    #
    #     return metadata

    def _ensure_directory_structure(self):
        if not os.path.isdir(self.img_dir):
            os.mkdir(self.img_dir)
        if not os.path.isdir(self.coords_dir):
            os.mkdir(self.coords_dir)

    def load_mungos(self, classes=None, by_page=False):
        """Loads all the available MuNG objects as a list. You need to make
        sure the objids don't clash across pages!"""
        self.update()
        if 'mung' not in self.views:
            raise SheetManagerDBError('Score {0}: mung view not available!'
                                      ''.format(self.name))
        mung_files = self.view_files('mung')

        mungos = []
        for f in mung_files:
            ms = parse_cropobject_list(f)
            if by_page:
                mungos.append(ms)
            else:
                mungos.extend(ms)

        if classes is not None:
            mungos = [m for m in mungos if m.clsname in classes]

        return mungos

    def get_ordered_notes(self, filter_tied=False, reverse_columns=False,
                          return_columns=False):
        """Returns the MuNG objects corresponding to notes in the canonical
        ordering: by page, system, left-to-right, and top-down within
        simultaneities (e.g. chords).

        :param reverse_columns: If set, will order the columns bottom-up
            instead of top-down. Use this for events alignment, not for score
            inference.
        """
        self.update()
        if 'mung' not in self.views:
            raise SheetManagerDBError('Score {0}: mung view not available!'
                                      ''.format(self.name))
        mung_files = self.view_files('mung')

        # Algorithm:
        #  - Create hard ordering constraints:
        #     - pages (already done: mungos_per_page)
        #     - systems

        notes_per_page = []

        for f in mung_files:
            mungos = parse_cropobject_list(f)
            mgraph = NotationGraph(mungos)
            _CONST = InferenceEngineConstants()

            note_mungos = [c for c in mungos
                           if 'midi_pitch_code' in c.data]
            system_mungos = [c for c in mungos if c.clsname == 'staff']
            system_mungos = sorted(system_mungos, key=lambda m: m.top)

            notes_per_system = []

            for s in system_mungos:
                system_notes = mgraph.ancestors(s,
                                                classes=_CONST.NOTEHEAD_CLSNAMES)
                for c in system_notes:
                    if 'midi_pitch_code' not in c.data:
                        print('Notehead without pitch: {0}'
                              ''.format(str(c)))
                        continue
                    if c.data['midi_pitch_code'] is None:
                        print('Notehead with pitch=None: {0}'
                              ''.format(str(c)))

                system_notes = [c for c in system_notes
                                if 'midi_pitch_code' in c.data]

                # print('Ancestors of system {0}: {1}'.format(s, system_notes))
                # Process simultaneities. We use a very small overlap ratio,
                # because we want to catch also chords that have noteheads
                # on both sides of the stem. Sorted top-down.
                # Remove all tied notes.
                if filter_tied:
                    system_notes = [m for m in system_notes
                                     if ('tied' not in m.data)
                                     or (('tied' in m.data) and (m.data['tied'] != 1))
                                    ]

                system_note_columns = group_mungos_by_column(system_notes,
                                                             MIN_OVERLAP_RATIO=0.05,
                                                             reverse_columns=reverse_columns)
                # print('System {0}: n_columns = {1}'
                #       ''.format(s.objid, len(system_note_columns)))
                ltr_sorted_columns = sorted(system_note_columns.items(),
                                            key=lambda kv: kv[0])
                # print('ltr_sorted_columns[0] = {0}'.format(ltr_sorted_columns[0]))
                system_ordered_simultaneities = [c[1]
                                                 for c in ltr_sorted_columns]
                # print('system_ordered_sims[0] = {0}'.format(system_ordered_simultaneities[0]))

                notes_per_system.append(system_ordered_simultaneities)

            # print('Total entries in notes_per_system = {0}'.format(len(notes_per_system)))
            notes_per_page.append(notes_per_system)


        # Data structure
        # --------------
        # notes_per_page = [
        #   notes_per_system_1 = [
        #       ordered_simultaneities = [
        #           simultaneity1 = [ a'', f'', c'', a' ],
        #           simultaneity2 = [ g'', e'', c'', bes' ],
        #           ...
        #       ]
        #   ],
        #   notes_per_system_2 = [
        #       simultaneity1 = [ ... ]
        #       ...
        #   ]
        # ]

        # Unroll simultaneities notes according to this data structure

        ### DEBUG
        # print('notes_per_page: {0}'.format(pprint.pformat(notes_per_page)))

        ordered_simultaneities = []
        for page in notes_per_page:
            for system in page:
                ordered_simultaneities.extend(system)

        if return_columns:
            return ordered_simultaneities

        ordered_notes = []
        for sim in ordered_simultaneities:
            ordered_notes.extend(list(reversed(sim)))

        return ordered_notes


def group_mungos_by_column(page_mungos, MIN_OVERLAP_RATIO=0.5,
                           reverse_columns=False):
    """Group symbols into columns.

    Two symbols are put in one column if their overlap is at least
    ``MIN_OVERLAP_RATIO`` of the left symbol.
    """
    _mdict = {m.objid: m for m in page_mungos}

    mungos_by_left = collections.defaultdict(list)
    for m in page_mungos:
        mungos_by_left[m.left].append(m)
    rightmost_per_column = {l: max([m.right for m in mungos_by_left[l]])
                            for l in mungos_by_left}
    mungo_to_leftmost = {m.objid: m.left for m in page_mungos}
    # Go greedily from left, take all objects that
    # overlap horizontally by half of the given column
    # width.
    lefts_sorted = sorted(mungos_by_left.keys())
    for i, l in list(enumerate(lefts_sorted))[:-1]:
        if mungos_by_left[l] is None:
            continue
        r = rightmost_per_column[l]
        mid_point = l + (r - l) * (1 - MIN_OVERLAP_RATIO)
        for l2 in lefts_sorted[i + 1:]:
            if l2 >= mid_point:
                break
            for m in mungos_by_left[l2]:
                mungo_to_leftmost[m.objid] = l
            mungos_by_left[l2] = None
    mungo_columns = collections.defaultdict(list)
    for objid in mungo_to_leftmost:
        l = mungo_to_leftmost[objid]
        mungo_columns[l].append(_mdict[objid])

    # ...sort the MuNG columns, from top to bottom or bottom-up
    # based on reverse_columns argument:
    sorted_mungo_columns = {l: sorted(mungos, key=lambda x: x.top,
                                      reverse=reverse_columns)
                            for l, mungos in mungo_columns.items()}
    return sorted_mungo_columns

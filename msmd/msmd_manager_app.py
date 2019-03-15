"""Sheet Manager is a program for extracting a multimodal set of views
of a piece of music from its encoding, such as a LilyPond or MusicXML file.
There is an audio view, a sheet music view, a MIDI view, and others may be
added.

A GUI for manually editing the sheet music view is started if no command-line
arguments are given.

Workflow
========

Each piece of music is represented as a directory. The source encoding file
(``*.ly`` or ``*.xml``; in this example we assume a LilyPond file is available) is
directly in that directory::

```
  bach-example-1/
    bach-example-1.ly
```

The PDF and MIDI representations (views) are generated directly from the source.
Additional views are then generated into sub-folders. There is an audio view
(or views, generated from MIDI using multiple sound-fonts and tempi), an image
view (one PNG file per page, generated from the PDF), an OMR view (locations
of relevant symbols and other structured information about the images of the
score; derived from the image view with the help of some LilyPond PDF tricks),
and a features (spec) view that represents the music as a feature matrix:
a spectrogram, an onset map, and a MIDI matrix.

Further details are in the documentation.

The ``msmd.py`` can be called either for batch processing as a script,
or a GUI for manually validating the annotations, especially alignment between
noteheads and onsets.
"""
from __future__ import print_function

import argparse
import logging
import pprint
import time
from PyQt5 import QtWidgets, uic

import os
import cv2
import glob
import pickle
import yaml
import sys
import numpy as np

# set backend to qt
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from muscima.io import export_cropobject_list, parse_cropobject_list
from msmd.alignments import group_mungos_by_system, \
    group_mungos_by_system_paths, \
    build_system_mungos_on_page, \
    align_score_to_performance, \
    alignment_stats, \
    is_aln_problem, \
    detect_system_regions_ly
from msmd.ly_parser import mung_midi_from_ly_links
from msmd.data_model.piece import Piece
from msmd.data_model.util import MSMDDBError
from msmd.utils import sort_by_roi, natsort, get_target_shape, corners2bbox
from msmd.pdf_parser import pdf2coords, parse_pdf
from msmd.colormaps import cmaps
import msmd.midi_parser as mp
from msmd.DEFAULT_CONFIG import SAMPLE_RATE, FPS, FRAME_SIZE, SPEC_FILTERBANK_NUM_BANDS, FMIN, FMAX


form_class = uic.loadUiType("gui/main.ui")[0]


# from omr.config.settings import DATA_ROOT as ROOT_DIR
# from omr.utils.data import MOZART_PIECES, BACH_PIECES, HAYDN_PIECES, BEETHOVEN_PIECES, CHOPIN_PIECES, SCHUBERT_PIECES, STRAUSS_PIECES
# PIECES = MOZART_PIECES + BACH_PIECES + HAYDN_PIECES + BEETHOVEN_PIECES + CHOPIN_PIECES + SCHUBERT_PIECES + STRAUSS_PIECES
# TARGET_DIR = "/home/matthias/mounts/home@rechenknecht1/Data/sheet_localization/real_music_sf"

# Audio augmentation settings:
#  - fixed SF and tempo combinations for no-aug. training and evaluation,
fixed_combinations = [(1.0, "ElectricPiano"),
                      (0.5, "grand-piano-YDP-20160804"),
                      (0.75, "grand-piano-YDP-20160804"),
                      (1.0, "grand-piano-YDP-20160804"),
                      (1.25, "grand-piano-YDP-20160804"),
                      (1.5, "grand-piano-YDP-20160804"),
                      (1.75, "grand-piano-YDP-20160804"),
                      (2.0, "grand-piano-YDP-20160804")]
#  - random soundfont and tempo selection for training data augmentation
tempo_ratios = [0.9, 0.95, 1.0, 1.05, 1.1]
sound_fonts = ["acoustic_piano_imis_1", "ElectricPiano", "YamahaGrandPiano"]


class MSMDManagerError(Exception):
    pass


class MSMDManagerGui(QtWidgets.QMainWindow, form_class):

    def __init__(self, parent):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

    def setup_bindings(self, mgr):
        """Sets up bindings between GUI elements and corresponding
        Sheet Manager actions."""

        # set up status bar
        self.status_label = QtWidgets.QLabel("")
        self.statusbar.addWidget(self.status_label)

        # connect to menu
        self.actionOpen_sheet.triggered.connect(mgr.open_sheet)
        self.pushButton_open.clicked.connect(mgr.open_sheet)

        self.comboBox_score.activated.connect(mgr.update_current_score)
        self.comboBox_performance.activated.connect(mgr.update_current_performance)

        # connect to omr menu
        self.actionInit.triggered.connect(mgr.init_omr)
        self.actionDetect_note_heads.triggered.connect(mgr.detect_note_heads)
        self.actionDetect_systems.triggered.connect(mgr.detect_systems)
        self.actionDetect_bars.triggered.connect(mgr.detect_bars)

        # connect to check boxes
        self.checkBox_showCoords.stateChanged.connect(mgr.plot_sheet)
        self.checkBox_showRois.stateChanged.connect(mgr.plot_sheet)
        self.checkBox_showSystems.stateChanged.connect(mgr.plot_sheet)
        self.checkBox_showBars.stateChanged.connect(mgr.plot_sheet)

        # Workflow for generating
        self.pushButton_mxml2midi.clicked.connect(mgr.mxml2midi)
        self.pushButton_ly2PdfMidi.clicked.connect(mgr.ly2pdf_and_midi)
        self.pushButton_pdf2Img.clicked.connect(mgr.pdf2img)
        self.pushButton_pdf2Coords.clicked.connect(mgr.pdf2coords)
        self.pushButton_renderAudio.clicked.connect(mgr.render_audio)
        self.pushButton_extractPerformanceFeatures.clicked.connect(
            mgr.extract_performance_features)
        self.pushButton_audio2sheet.clicked.connect(mgr.match_audio2sheet)

        self.pushButton_ClearState.clicked.connect(mgr.reset)

        # Editing coords
        self.pushButton_editCoords.clicked.connect(mgr.edit_coords)
        self.pushButton_loadSheet.clicked.connect(mgr.load_sheet)
        self.pushButton_loadPerformanceFeatures.clicked.connect(
            mgr.load_performance_features)
        self.pushButton_loadCoords.clicked.connect(mgr.load_coords)
        self.pushButton_saveCoords.clicked.connect(mgr.save_coords)

        self.spinBox_window_top.valueChanged.connect(mgr.update_staff_windows)
        self.spinBox_window_bottom.valueChanged.connect(mgr.update_staff_windows)
        self.spinBox_page.valueChanged.connect(mgr.edit_coords)

        # Deprecated in favor of batch processing
        # self.pushButton_renderAllAudios.clicked.connect(mgr.render_all_audios)
        # self.pushButton_copySheets.clicked.connect(mgr.copy_sheets)
        # self.pushButton_prepareAll.clicked.connect(mgr.prepare_all_audio)
        # self.pushButton_parseAllMidis.clicked.connect(mgr.parse_all_midis)

        # Params for sheet editing: system bbox --> roi
        mgr.window_top = self.spinBox_window_top.value()
        mgr.window_bottom = self.spinBox_window_bottom.value()


class MSMDManager(object):
    """Processing and GUI for the Multi-Modal MIR dataset.

    The workflow is wrapped in the ``process_piece()`` method.
    """

    def __init__(self, parent=None, interactive=True):
        """
        Constructor
        """
        self.window_bottom = 5
        self.window_top = 5

        self.target_width = 835
        self.retain_audio = True  # By default, the manager generates everything
        self.n_augmentation_performances = 7

        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        # Refactoring to use the piece-performance-score data model
        self.piece = None
        self.current_performance = None
        self.current_score = None

        # Old interface...
        self.folder_name = None
        self.piece_name = None

        # Piece encoding files
        self.lily_file = None
        self.mxml_file = None
        self.midi_file = None

        self.lily_normalized_file = None

        # Score Elements Editor
        self.fig = None
        self.fig_manager = None
        self.click_0 = None
        self.click_1 = None
        self.press = False
        self.drawObjects = []

        self.sheet_pages = None
        self.page_coords = None
        self.page_systems = None
        self.page_rois = None
        self.page_bars = None
        self.page_mungos = None

        # Current score properties
        self.score_name = None
        self.pdf_file = None
        self.sheet_folder = None
        self.coord_folder = None

        self.omr = None

        self.axis_label_fs = 16

        # Current performance properties
        self.performance_name = None
        # Expected features
        self.midi_matrix = None
        self.spec = None
        self.onsets = None
        self.note_events = None

        self._page_mungo_centroids = None
        self._page_centroids2mungo_map = None

        self.score_performance_alignment = None
        # Dict: objid --> event_idx

        self.interactive = interactive

        if self.interactive:
            print('MSMDManager: running in INTERACTIVE mode')
            self.gui = MSMDManagerGui(parent=parent)
            self.gui.setup_bindings(self)
        else:
            self.gui = None
            print('MSMDManager: running in BATCH mode')

    def reset(self):
        """Resets the MSMDManager back to its initial state."""
        self.piece = None
        self.current_score = None
        self.current_performance = None
        self.folder_name = None
        self.piece_name = None
        self.lily_file = None
        self.mxml_file = None
        self.midi_file = None
        self.lily_normalized_file = None
        self.fig = None
        self.fig_manager = None
        self.click_0 = None
        self.click_1 = None
        self.press = False
        self.drawObjects = []
        self.sheet_pages = None
        self.page_coords = None
        self.page_systems = None
        self.page_rois = None
        self.page_bars = None
        self.page_mungos = None
        self.score_name = None
        self.pdf_file = None
        self.sheet_folder = None
        self.coord_folder = None
        self.omr = None
        self.axis_label_fs = 16
        self.performance_name = None
        self.midi_matrix = None
        self.spec = None
        self.onsets = None
        self.note_events = None
        self._page_mungo_centroids = None
        self._page_centroids2mungo_map = None
        self.score_performance_alignment = None

    def open_sheet(self):
        """Choose a piece directory to open through a dialog window.
        """

        # piece root folder
        folder_name = str(QtWidgets.QFileDialog.getExistingDirectory(
            self.gui,
            "Select Sheet Music",
            "."))
        self.load_piece(folder_name=folder_name)

    def process_piece(self, piece_folder, workflow="ly", omr=False):
        """Process the entire piece, using the specified workflow.

        :param piece_folder: The path of the requested piece folder.
            If the piece folder is not found, raises a ``MSMDManagerError``.

        :param workflow: Which workflow to use, based on the available encoding
            of the piece. By default, uses ``"ly"``, the LilyPond workflow.
            (Currently, only the LilyPond workflow is implemented.)
        """
        if not os.path.isdir(piece_folder):
            raise MSMDManagerError('Piece folder not found: {0}'
                                   ''.format(piece_folder))

        is_workflow_ly = (workflow == "ly") or (workflow == "Ly")
        if not is_workflow_ly:
            raise MSMDManagerError('Only the LilyPond workflow is currently '
                                   ' supported!'
                                   ' Use arguments workflow="ly" or "Ly". '
                                   ' (Got: workflow="{0}")'.format(workflow))

        self.reset()

        self.load_piece(piece_folder)

        if is_workflow_ly and not self.lily_file:
            raise MSMDManagerError('Piece does not have LilyPond source'
                                   ' available; cannot process with workflow={0}'
                                   ''.format(workflow))

        # Load
        print('RUNNING: ly2pdf_and_midi()')
        self.ly2pdf_and_midi()

        print('RUNNING: pdf2img()')
        self.pdf2img()

        # Exploiting LilyPond point-and-click PDF annotations to get noteheads
        if is_workflow_ly:
            print('RUNNING: pdf2coords')
            self.pdf2coords()

        # Render Audio performances (also augmented versions)
        self.render_audio()

        # For each performance, do the DTW-alignment and save to mung
        for performance in self.piece.load_all_performances(require_audio=False,
                                                            require_midi=True):
            self.current_performance = performance
            self.extract_performance_features(retain_audio=self.retain_audio)

            # Alignment
            self.load_performance_features()
            self.load_sheet()

        # OMR
        if omr:
            if not self.omr:
                self.init_omr()

            self.detect_systems()
            self.detect_bars()

            # If we are unlucky and have no LilyPond source:
            if not is_workflow_ly:
                self.detect_note_heads()

        # Align written notes and performance onsets
        # self.sort_bar_coords()
        # self.sort_note_coords()

        ############################
        # Detect systems
        for page_id in range(self.n_pages):
            self.detect_systems_on_page(page_id, with_omr=False)

        self.save_coords()

        # Status report
        page_stats, piece_stats = self.collect_stats()

        # Log to metadata
        self.piece.metadata = dict()
        self.piece.metadata['n_pages'] = self.n_pages
        self.piece.metadata['n_performances'] = len(self.piece.performances)
        self.piece.metadata['n_scores'] = len(self.piece.scores)
        self.piece.metadata['aln_piece_stats'] = dict([kv for kv in piece_stats._asdict().items()
                                                       if isinstance(kv[1], int)])
        self.piece.metadata['aln_page_stats'] = [dict([kv for kv in s._asdict().items()
                                                       if isinstance(kv[1], int)])
                                                 for s in page_stats.values()]
        self.piece.dump_metadata()

        return page_stats, piece_stats

    def get_stats_of_piece(self, piece_folder, require_alignment=False):
        """Assuming the given piece is finished, compute the stats."""
        self.reset()

        if not os.path.isdir(piece_folder):
            raise MSMDManagerError('Piece folder not found: {0}'
                                   ''.format(piece_folder))
        self.load_piece(piece_folder)
        self.load_performance_features()
        self.load_sheet(update_alignment=False)
        if len(self.score_performance_alignment) == 0:
            print('WARNING: computing stats of a piece which does not have'
                  ' anything aligned!')
            if require_alignment:
                print('Updating alignment...')
                self.load_sheet(update_alignment=True)

        page_stats, piece_stats = self.collect_stats()
        return page_stats, piece_stats

    def collect_stats(self):
        """Collects various statistics about the piece."""
        page_stats = {}
        all_mungos = []
        all_events = self.note_events
        for page, mungos in enumerate(self.page_mungos):
            all_mungos.extend(mungos)
            events = self.page_note_events(page)

            stats = alignment_stats(mungos, events,
                                    self.score_performance_alignment)
            page_stats[page] = stats

        # Global statistics
        piece_stats = alignment_stats(all_mungos, all_events,
                                      self.score_performance_alignment)

        # print('Collected piece stats: ', piece_stats)
        # print('All mungos: {0}, all events: {1}, aln: {2}'
        #       ''.format(len(all_mungos), len(all_events), len(self.score_performance_alignment)))

        return page_stats, piece_stats

    def page_note_events(self, page):
        """Returns the list of note events that we know should belong on
        the given page, from the alignment. (This means that events at page
        breaks might not be taken into account.)"""
        mungos = self.page_mungos[page]

        note_events = self.note_events
        first_onset = np.inf
        last_onset = 0

        for m in mungos:
            if m.objid in self.score_performance_alignment:
                e_idx = self.score_performance_alignment[m.objid]
                e = note_events[e_idx]
                e_onset = e[0]
                if e_onset < first_onset:
                    first_onset = e_onset
                if e_onset > last_onset:
                    last_onset = e_onset

        page_event_dict = {e_idx: e for e_idx, e in enumerate(note_events)
                           if first_onset <= e[0] <= last_onset}
        return page_event_dict

    # r-x GUI
    def load_piece(self, folder_name):
        """Given a piece folder, set the current state of MSMDManager to this piece.
        If the folder does not exist, does nothing."""
        if not os.path.isdir(folder_name):
            print('Loading piece from folder {0} failed: folder does not exist!'
                  ''.format(folder_name))
            return
        # Normalize
        folder_name = os.path.normpath(folder_name)
        self.folder_name = folder_name

        # Create the current Piece
        piece_name = os.path.basename(self.folder_name)
        collection_root = os.path.split(self.folder_name)[0]
        workflow = 'ly'

        piece = Piece(name=piece_name,
                      root=collection_root,
                      authority_format=workflow)
        self.piece = piece

        # Re-done as not GUI; also updates perf. selection
        self._refresh_score_and_performance_selection()

        if self.gui:
            # This is already done in _refresh_score_and_performance_selection()
            # self.update_current_performance() # @@ GUI
            # self.update_current_score() # @@ GUI

            if self.pdf_file is not None:
                self.gui.lineEdit_pdf.setText(self.pdf_file)
            else:
                self.gui.lineEdit_pdf.setText("")

        # Old workflow.
        self.piece_name = piece_name
        self.folder_name = piece.folder

        # compile encoding file paths
        self.mxml_file = self.piece.encodings.get('mxml', "")
        self.lily_file = self.piece.encodings.get('ly', "")
        self.midi_file = self.piece.encodings.get('midi', "")

        self.lily_normalized_file = self.piece.encodings.get('norm.ly', "")

        # update gui elements
        if self.gui:
            self.gui.lineEdit_mxml.setText(self.mxml_file)
            self.gui.lineEdit_lily.setText(self.lily_file)
            self.gui.lineEdit_midi.setText(self.midi_file)

    # r-- GUI
    def update_current_score(self):
        """Reacts to a change in the score that MSMDManager is supposed
        to be currently processing."""
        score_name = str(self.gui.comboBox_score.currentText())
        self.set_current_score(score_name)

    def set_current_score(self, score_name):
        """Set the current Score."""
        if (score_name is None) or (score_name == ""):
            logging.info('Selection provided no score; probably because'
                         ' no scores are available.')
            return

        try:
            current_score = self.piece.load_score(score_name)
        except MSMDDBError as e:
            print('Could not load score {0}: malformed?'
                  ' Error message: {1}'.format(score_name, e))
            return

        self.score_name = score_name
        self.current_score = current_score

        self.pdf_file = self.current_score.pdf_file
        self.sheet_folder = self.current_score.img_dir
        self.coord_folder = self.current_score.coords_dir

    # r-- GUI
    def update_current_performance(self):
        perf_name = str(self.gui.comboBox_performance.currentText())
        self.set_current_performance(perf_name)

    def set_current_performance(self, perf_name):
        if (perf_name is None) or (perf_name == ""):
            logging.info('Selection provided no performance; probably because'
                         ' no performances are available.')
            return

        try:
            current_performance = self.piece.load_performance(perf_name,
                                                              require_audio=False,
                                                              require_midi=True)
        except MSMDDBError as e:
            print('Could not load performance {0}: malformed?'
                  ' Error message: {1}'.format(perf_name, e))
            return

        self.current_performance = current_performance
        self.performance_name = perf_name

    # rwx GUI
    def _refresh_score_and_performance_selection(self):
        """Synchronizes the selection of scores and performances.
        Tries to retain the previous score/performance."""
        if not self.gui:
            try:
                available_scores = self.piece.available_scores
                if len(available_scores) > 0:
                    self.set_current_score(available_scores[0])
            except Exception:
                print('Cannot update score: no score is available!')
            try:
                available_performances = self.piece.available_performances
                if len(available_performances) > 0:
                    self.set_current_performance(available_performances[0])
            except Exception:
                print('Cannot update performance: no performance is available!')
            return

        # With GUI:

        old_score_idx = self.gui.comboBox_score.currentIndex()
        old_score = str(self.gui.comboBox_score.itemText(old_score_idx))

        old_perf_idx = self.gui.comboBox_performance.currentIndex()
        old_perf = str(self.gui.comboBox_performance.itemText(old_perf_idx))

        self.piece.update()

        self.gui.comboBox_score.clear()
        self.gui.comboBox_score.addItems(self.piece.available_scores)
        if old_score in self.piece.available_scores:
            idx = self.piece.available_scores.index(old_score)
            self.gui.comboBox_score.setCurrentIndex(idx)
        else:
            self.gui.comboBox_score.setCurrentIndex(0)
            self.update_current_score()

        self.gui.comboBox_performance.clear()
        self.gui.comboBox_performance.addItems(self.piece.available_performances)
        if old_perf in self.piece.available_performances:
            idx = self.piece.available_performances.index(old_perf)
            self.gui.comboBox_performance.setCurrentIndex(idx)
        else:
            self.gui.comboBox_performance.setCurrentIndex(0)
            self.update_current_performance()

    # -wx GUI
    def ly2pdf_and_midi(self, quiet=False):
        """Convert the LilyPond file to PDF and MIDI (which is done automatically, if the Ly
        file contains the \midi { } directive)."""
        self.normalize_ly()

        if self.gui:
            self.gui.status_label.setText("Convert Ly to pdf ...")

        if not os.path.isfile(self.lily_normalized_file):
            self.gui.status_label.setText("done! (Error: LilyPond file not found!)")
            print('Error: LilyPond file not found!')
            return

        # Set PDF paths. LilyPond needs the output path without the .pdf suffix
        # This creates the default PDF.
        pdf_path_nosuffix = os.path.join(self.folder_name, self.piece_name)
        pdf_path = pdf_path_nosuffix + '.pdf'

        cmd_base = 'lilypond'
        cmd_point_and_click = ' -e"(ly:set-option \'point-and-click \'(note-event))"'
        cmd_suppress_midi = '-e"(set! write-performances-midis (lambda (performances basename . rest) 0))"'

        # Do not overwrite an existing MIDI file.
        suppress_midi = False
        if os.path.isfile(self.midi_file):
            suppress_midi = True

        cmd_options = cmd_point_and_click
        if suppress_midi:
            cmd_options += ' {0}'.format(cmd_suppress_midi)

        cmd = cmd_base + ' -o {0} '.format(pdf_path_nosuffix) \
              + cmd_options \
              + ' {0}'.format(self.lily_normalized_file)

        if quiet:
            cmd += ' &> /dev/null'

        # Run LilyPond here.
        os.system(cmd)

        # If successful, the PDF file will be there:
        if os.path.isfile(pdf_path):
            print('...Adding score to piece')
            self.piece.add_score(name=self.piece.default_score_name,
                                 pdf_file=pdf_path,
                                 overwrite=True)

            print('...Setting current score')
            self.set_current_score(self.piece.default_score_name)

            if self.gui:
                self.gui.lineEdit_pdf.setText(self.pdf_file)

            # Cleanup!
            os.unlink(pdf_path)

        else:
            print('Warning: LilyPond did not generate PDF file.'
                  ' Something went badly wrong.')

        # Check if the MIDI file was actually created.
        print('Adding MIDI file...')
        output_midi_file = os.path.join(self.folder_name, self.piece_name) + '.mid'
        if not os.path.isfile(output_midi_file):
            output_midi_file += 'i'  # If it is not *.mid, maybe it has *.midi
            if not os.path.isfile(output_midi_file):
                print('Warning: LilyPond did not generate corresponding MIDI file. Check *.ly source'
                      ' file for \\midi { } directive.')
            else:
                self.midi_file = output_midi_file
                if self.gui:
                    self.gui.lineEdit_midi.setText(self.midi_file)

        # Update with the MIDI encoding
        print('Updating piece')
        self.piece.update()

        print('Refreshing score and perf. selection')
        self._refresh_score_and_performance_selection()

    def normalize_ly(self, quiet=False):
        """ Converts lilypond file to absolute. There is a sanity check:
        if there are too many consecutive apostrophes in the output file,
        we assume that the file was already encoded as absolute, and keep
        it without changes.

        The backlinks from ly --> pdf generation lead to the **normalized**
        LilyPond file!
        """
        if not os.path.exists(self.lily_file):
            print('Cannot normalize, LilyPond file is missing!')
            return

        lily_normalized_fname = os.path.splitext(self.lily_file)[0] + '.norm.ly'
        self.lily_normalized_file = os.path.join(self.piece.folder,
                                                 lily_normalized_fname)
        os.system('cp {0} {1}'.format(self.lily_file, self.lily_normalized_file))

        _quiet = ""
        if quiet:
            _quiet = " &> /dev/null"

        if not os.path.exists(self.lily_normalized_file):
            print('Initializing normalized LilyPond file {0} failed!')
            return

        # Update to current LilyPond version
        convert_cmd = "convert-ly -e {0}" \
                      "".format(self.lily_normalized_file) + _quiet
        logging.info('Normalizing LilyPond: updating syntax to latest possible'
                     ' version: {0}'.format(convert_cmd))
        os.system(convert_cmd)

        # Translate to default pitch language?
        translate_cmd = 'ly -i "translate english" {0}' \
                        ''.format(self.lily_normalized_file) + _quiet
        logging.info('Normalizing LilyPond: translating pitch names: {0}'
                     ''.format(translate_cmd))
        os.system(translate_cmd)

        # Convert to absolute
        base_cmd = 'ly rel2abs'
        cmd = base_cmd + ' -i {0}'.format(self.lily_normalized_file) + _quiet

        logging.info('Normalizing LilyPond: moving to absolute: {0}'.format(cmd))
        os.system(cmd)

        # Check if the normalization didn't produce an absurd number of apostrophes.
        # That would be a sign of having interpreted an absolute file as relative.
        with open(self.lily_normalized_file) as hdl:
            ly_str = ''.join([l for l in hdl])

        LY_SANITY_CHECK_TOKEN = "''''''"
        if LY_SANITY_CHECK_TOKEN in ly_str:
            print('...found five apostrophes in normalized LilyPond file,'
                  ' which should not happen in absolute-encoded music.'
                  ' We probably already had an absolute file. Returning back'
                  ' to original file.')
            os.system('cp {0} {1}'.format(self.lily_file, self.lily_normalized_file))

        # Remove backup of normalized file
        _ly_norm_backup_file = self.lily_normalized_file + '~'
        if os.path.isfile(_ly_norm_backup_file):
            os.unlink(_ly_norm_backup_file)

        # Update the encodings dict to include the *.norm.ly file
        self.piece.update()

    # r-x GUI
    def pdf2img(self, quiet=False):
        """ Convert pdf file to image """
        _quiet = ""
        if quiet:
            _quiet = " &> /dev/null"

        # self.gui.status_label.setText("Convert pdf to images ...")

        os.system("rm tmp/*.png" + _quiet)
        pdf_path = self.current_score.pdf_file
        # pdf_path = os.path.join(self.folder_name,
        #                         self.piece_name +
        #                         self.score_name + '.pdf')
        cmd = "convert -density 150 {0} -quality 90 tmp/page.png" \
              "".format(pdf_path) + _quiet
        os.system(cmd)

        if self.gui:
            self.gui.status_label.setText("Resizing images ...")
        img_dir = self.current_score.img_dir
        self.current_score.clear_images()

        file_paths = glob.glob("tmp/*.png")
        file_paths = natsort(file_paths)
        for i, img_path in enumerate(file_paths):
            img = cv2.imread(img_path, -1)

            # compute resize stats
            target_height, target_width = get_target_shape(img, self.target_width)
            #
            # ratio = float(self.target_width) / img.shape[1]
            # target_height = img.shape[0] * ratio
            #
            # target_width = int(target_width)
            # target_height = int(target_height)

            img_rsz = cv2.resize(img, (target_width, target_height))

            out_path = os.path.join(img_dir, "%02d.png" % (i + 1))
            cv2.imwrite(out_path, img_rsz)

            if self.gui:
                if self.gui.checkBox_showSheet.isChecked():
                    plt.figure()
                    plt.title('{0}, page {1}'.format(self.piece_name, i+1))
                    plt.imshow(img_rsz, cmap=plt.cm.gray)

        if self.gui:
            if self.gui.checkBox_showSheet.isChecked():
                plt.show()

        # self.gui.status_label.setText("done!")

    # rwx GUI
    def pdf2coords(self):
        """Extracts notehead centroid coords and MuNG features
        from the PDF of the current Score. Saves them to the ``coords/``
        and ``mung/`` view."""
        logging.info("Extracting coords from pdf ...")

        # self.status_label.setText("Extracting coords from pdf ...")

        if not os.path.exists(self.pdf_file):
            # self.status_label.setText("Extracting coords from pdf failed: PDF file not found!")
            logging.warning("Extracting coords from pdf failed: PDF file not found: {0}"
                            "".format(self.pdf_file))
            return

        self.load_sheet()  # update_alignment=False)

        n_pages = len(self.page_coords)
        if n_pages == 0:
            # self.status_label.setText("Extracting coords from pdf failed: could not find sheet!")

            logging.warning("Extracting coords from pdf failed: could not find sheet!"
                            " Generate the image files first.")
            return

        # Run PDF coordinate extraction
        centroids, bboxes, mungos = parse_pdf(self.pdf_file,
                                              target_width=self.target_width,
                                              with_links=True,
                                              collection_name=os.path.basename(self.piece.collection_root),
                                              score_name=self.current_score.name)
        # centroids = pdf2coords(self.pdf_file, target_width=self.target_width)

        # Check that the PDF corresponds to the generated images...
        if len(centroids) != n_pages:
            logging.warning("Something is wrong with the PDF vs. the generated"
                            " images: page count"
                            " does not match (PDF parser: {0} pages,"
                            " images: {1}). Re-generate the images from PDF."
                            "".format(len(centroids), n_pages))
            return

        # Derive no. of pages from centroids
        for page_id in centroids:
            logging.info('Page {0}: {1} note events found'
                         ''.format(page_id, centroids[page_id].shape[0]))
            self.page_coords[page_id] = centroids[page_id]

        # Save the coordinates
        self.save_note_coords()

        # refresh view
        self.sort_note_coords()

        # update sheet statistics
        self.update_sheet_statistics()

        # Add MuNG MIDI pitch codes
        mungos = {page: mung_midi_from_ly_links(mungos[page])
                  for page in mungos}

        system_bboxes = {}
        system_mungo_groups = {}
        _system_start_objid = max([max([m.objid for m in ms])
                                   for ms in mungos.values()]) + 1
        logging.info('System objids will start from: {0}'
                     ''.format(_system_start_objid))

        for i, page in enumerate(mungos.keys()):
            logging.info('Page {0}: total MuNG objects: {1}, with pitches: {2}'
                         ''.format(page, len(mungos[page]),
                                   len([m for m in mungos[page]
                                        if 'midi_pitch_code' in m.data])))
            page_system_bboxes, page_system_mungo_groups = \
                group_mungos_by_system_paths(
                    page_mungos=mungos[page],
                    score_img=self.sheet_pages[i],
                    page_num=i+1
                )
            system_bboxes[page] = page_system_bboxes
            system_mungo_groups[page] = page_system_mungo_groups

            # Build the MuNG objects for systems and link
            # the others.
            system_mungos = build_system_mungos_on_page(
                page_system_bboxes,
                page_system_mungo_groups,
                start_objid=_system_start_objid
            )
            _system_start_objid = max([sm.objid for sm in system_mungos]) + 1

            combined_mungos = list(mungos[page]) + system_mungos
            mungos[page] = combined_mungos

        self.page_systems = [[] for _ in range(n_pages)]
        for page, bboxes in system_bboxes.items():
            corners = []
            logging.info('Page {0}: system bboxes = {1}'.format(page, bboxes))
            for t, l, b, r in bboxes:
                corners.append([[t, l], [t, r], [b, r], [b, l]])
            corners_np = np.asarray(corners)
            logging.debug('Corners shape: {0}'.format(corners_np.shape))
            self.page_systems[page] = corners_np

        # Save systems & refresh
        self.save_system_coords()
        self.sort_note_coords()
        self.update_sheet_statistics()

        # Save MuNG
        mung_strings = {page: export_cropobject_list(mungos[page])
                        for page in range(len(mungos))}
        self.current_score.add_paged_view('mung', mung_strings,
                                          file_fmt='xml',
                                          binary=False,
                                          prefix=None,
                                          overwrite=True)

        if self.gui:
            if self.gui.checkBox_showExtractedCoords.isChecked():
                # Maybe we already have the ROIs for this image.
                self.load_coords()
                self.sort_note_coords()
                self.update_sheet_statistics()
                self.edit_coords()

        # self.status_label.setText("done!")

    # -w- GUI
    def mxml2midi(self):
        """
        Convert mxml to midi file. Uses LilyPond as an intermediary.
        """

        # generate midi
        os.system("musicxml2ly --midi -a %s -o tmp/tmp.ly" % self.mxml_file)
        os.system("lilypond -o tmp/tmp tmp/tmp.ly")

        # copy midi file
        self.midi_file = os.path.join(self.folder_name, self.piece_name + '.midi')
        os.system("cp tmp/tmp.midi %s" % self.midi_file)
        if self.gui:
            self.gui.lineEdit_midi.setText(self.midi_file)

    def render_audio(self, performance_prefix="", clear_previous_performances=False):
        """
        Render audio from midi.
        """
        # self.status_label.setText("Rendering audio ...")

        if 'midi' not in self.piece.encodings:
            raise MSMDManagerError('Cannot render audio from current piece:'
                                   ' no MIDI encoding available!')

        if not performance_prefix:
            performance_prefix = self.piece.name

        # TODO: think about this
        # random set of audio augmentations
        combinations = list(fixed_combinations)
        while len(combinations) < len(fixed_combinations) + self.n_augmentation_performances:
            comb = (np.random.choice(tempo_ratios[:]), np.random.choice(sound_fonts[:]))
            if comb in combinations:
                continue
            combinations.append(comb)

        if clear_previous_performances:
            self.piece.clear_performances()

        # for ratio in tempo_ratios:
        #     for sound_font in sound_fonts:
        for ratio, sound_font in combinations:
            performance_name = performance_prefix \
                               + '_tempo-{0}'.format(int(1000 * ratio)) \
                               + '_{0}'.format(sound_font)

            audio_file, perf_midi_file = mp.render_audio(self.piece.encodings['midi'],
                                                         sound_font=sound_font,
                                                         tempo_ratio=ratio, velocity=64)

            # copy all relevant files and information to piece object
            self.piece.add_performance(name=performance_name,
                                       audio_file=audio_file,
                                       midi_file=perf_midi_file,
                                       overwrite=True)

            self.piece.update()
            self._refresh_score_and_performance_selection()

            # Cleanup tmp-files
            os.unlink(audio_file)
            os.unlink(perf_midi_file)

        # self.status_label.setText("done!")

    # r-- GUI
    def extract_performance_features(self, retain_audio=False):
        """
        Parse all performance midi files and create the corresponding
        feature files: spectrogram, MIDI matrix, and onsets. Assumes both
        the performance MIDI and the audio file are available; if performance
        MIDI is not available, skips the performance. (This is behavior that
        needs to be changed with refactoring the MIDI parser to take care
        of the spectrogram separately from the MIDI-based features, in order
        to have any useful runtime feature extraction for performance test
        data!)
        """

        # For each performance:
        #  - load performance MIDI
        #  - compute onsets
        #  - compute MIDI matrix
        #  - save onsets as perf. feature
        #  - save MIDI matrix as perf.feature
        #  - load performance audio
        #  - compute performance spectrogram
        #  = save spectrogram as perf. feature
        print('Feature Extraction on midi and audio files...')

        performance = self.current_performance

        if performance.audio is None:
            logging.info('Performance {0} has no audio, skipping'
                         ' performance feature extraction.'
                         ''.format(performance.name))

        audio_file_path = performance.audio
        midi_file_path = performance.midi
        if not os.path.isfile(midi_file_path):
            logging.warning('Performance {0} has no MIDI file, cannot'
                            ' compute onsets and MIDI matrix. Skipping.'
                            ''.format(performance.name))

        midi_parser = mp.MidiParser(show=(self.gui and self.gui.checkBox_showSpec.isChecked()))
        spectrogram, onsets, durations, midi_matrix, note_events = midi_parser.process(
            midi_file_path, audio_file_path, return_midi_matrix=True,
            frame_size=FRAME_SIZE, sample_rate=SAMPLE_RATE, fps=FPS, num_bands=SPEC_FILTERBANK_NUM_BANDS, fmin=FMIN, fmax=FMAX)

        performance.add_feature(spectrogram, 'spec.npy', overwrite=True)
        performance.add_feature(onsets, 'onsets.npy', overwrite=True)
        performance.add_feature(durations, 'durations.npy', overwrite=True)
        performance.add_feature(midi_matrix, 'midi.npy', overwrite=True)
        performance.add_feature(note_events, 'notes.npy', overwrite=True)

        self.onsets = onsets
        self.durations = durations
        self.midi_matrix = midi_matrix
        self.spec = spectrogram
        self.note_events = note_events

        if not retain_audio:
            os.unlink(audio_file_path)

        # set number of onsets in gui
        if self.gui:
            print(self.onsets)
            self.gui.lineEdit_nOnsets.setText(str(len(self.onsets)))

        # self.status_label.setText("done!")

    # -w- GUI
    def load_performance_features(self):
        """
        Load spectrogram, MIDI matrix and onsets from the current performance.
        """
        if self.current_performance is None:
            logging.warning('Cannot load performance features without'
                            ' selecting a performance!')
            return

        try:
            midi_matrix = self.current_performance.load_midi_matrix()
        except MSMDDBError as e:
            logging.warning('Loading midi matrix from current performance {0}'
                            ' failed: {1}'.format(self.current_performance.name,
                                                  e))
            return
        self.midi_matrix = midi_matrix

        try:
            onsets = self.current_performance.load_onsets()
        except MSMDDBError as e:
            logging.warning('Loading onsets from current performance {0}'
                            ' failed: {1}'.format(self.current_performance.name,
                                                  e))
            return
        self.onsets = onsets

        try:
            spectrogram = self.current_performance.load_spectrogram()
        except MSMDDBError as e:
            logging.warning('Loading spectrogram from current performance {0}'
                            ' failed: {1}'.format(self.current_performance.name,
                                                  e))
            return
        self.spec = spectrogram

        try:
            notes = self.current_performance.load_note_events()
        except MSMDDBError as e:
            logging.warning('Loading note events from current performance {0}'
                            ' failed: {1}'.format(self.current_performance.name,
                                                  e))
            return
        self.note_events = notes

        # set number of onsets in gui
        if self.gui:
            self.gui.lineEdit_nOnsets.setText(str(len(self.onsets)))

    # -w- GUI
    def load_sheet(self, update_alignment=True):
        """Load sheet images of current piece to prepare for OMR
        and/or coords editing.
        """
        # self.status_label.setText("Loading sheet ...")

        self.sheet_pages = []
        self.page_coords = []
        self.page_rois = []
        self.page_systems = []
        self.page_bars = []

        # prepare paths
        img_files = self.current_score.image_files

        # initialize page data (currently loads just empty coords)
        n_pages = len(img_files)
        for i in range(n_pages):
            self.sheet_pages.append(cv2.imread(img_files[i], 0))

            self.page_coords.append(np.zeros((0, 2)))
            self.page_rois.append([])
            self.page_systems.append(np.zeros((0, 4, 2)))
            self.page_bars.append(np.zeros((0, 2, 2)))

        if self.gui:
            self.gui.spinBox_page.setMaximum(n_pages - 1)

        self.update_sheet_statistics()

        # Loads the coordinates, if there are any stored
        self.load_coords()

        # Load MuNG objects, if there are any
        if 'mung' in self.current_score.views:
            self.load_mung(update_alignment=update_alignment)

        # self.status_label.setText("done!")

    def load_mung(self, update_alignment=True):
        """ Loads the Notation Graph representation. """
        if 'mung' not in self.current_score.views:
            print('Loading MuNG failed: notation graph not available!')
            return

        mung_files = self.current_score.view_files('mung')

        page_mungos = [parse_cropobject_list(mf) for mf in mung_files]
        self.page_mungos = page_mungos

        # Serves as an interface between positions in the image
        # and the MuNG objects.
        self._page_centroids2mungo_map = []
        self._page_mungo_centroids = []
        for mungos in self.page_mungos:
            mungo_centroids = [((m.top + m.bottom) / 2., (m.left + m.right) / 2)
                               for m in mungos]
            mungo_centroids_map = {c: m for c, m in zip(mungo_centroids,
                                                        mungos)}
            self._page_centroids2mungo_map.append(mungo_centroids_map)
            self._page_mungo_centroids.append(np.asarray(mungo_centroids))

        logging.info('Page mungo centroids map sizes per page: {0}'
                     ''.format([len(self._page_centroids2mungo_map[i])
                                for i in range(len(self.page_mungos))]))

        if update_alignment:
            self.update_mung_alignment()
        else:
            self.load_mung_alignment()

    def load_coords(self):
        """ Load coordinates """
        self.load_system_coords()
        self.load_bar_coords()
        self.load_note_coords()

        self.update_sheet_statistics()

    def load_system_coords(self):
        """ Load system coordinates """
        # self.status_label.setText("Loading system coords ...")

        # prepare paths
        coord_dir = self.current_score.coords_dir
        coord_files = np.sort(glob.glob(coord_dir + "/systems_*.npy"))

        # load data
        if len(coord_files) > 0:
            for i in range(self.n_pages):
                if os.path.exists(coord_files[i]):
                    self.page_systems[i] = np.load(coord_files[i])

        # convert systems to rois
        self.systems_to_rois()

        # self.status_label.setText("done!")

    def load_bar_coords(self):
        """ Load bar coordinates """
        # self.status_label.setText("Loading system coords ...")

        # prepare paths
        coord_dir = self.current_score.coords_dir
        coord_files = np.sort(glob.glob(coord_dir + "/bars_*.npy"))

        # load data
        if len(coord_files) > 0:
            for i in range(self.n_pages):
                if os.path.exists(coord_files[i]):
                    self.page_bars[i] = np.load(coord_files[i])

        self.sort_bar_coords()

        # self.status_label.setText("done!")

    def load_note_coords(self):
        """ Load note coordinates """
        # self.status_label.setText("Loading coords ...")

        # prepare paths
        coord_dir = self.current_score.coords_dir
        coord_files = np.sort(glob.glob(coord_dir + "/notes_*.npy"))

        # load data
        if len(coord_files) > 0:
            for i in range(self.n_pages):
                if os.path.exists(coord_files[i]):
                    page_coords = np.load(coord_files[i])
                    self.page_coords[i] = page_coords

        self.sort_note_coords()

        # self.status_label.setText("done!")

    def systems_to_rois(self):
        """ Convert systems to rois"""
        self.page_rois = []

        # sort systems
        for i in range(self.n_pages):
            sorted_idx = np.argsort(self.page_systems[i][:, 0, 0])
            self.page_systems[i] = self.page_systems[i][sorted_idx]

        for i in range(self.n_pages):
            width = self.sheet_pages[i].shape[1]
            self.page_rois.append([])
            for system_coords in self.page_systems[i]:
                r_min = system_coords[0, 0] - self.window_top
                r_max = system_coords[3, 0] + self.window_bottom
                topLeft = [r_min, 0]
                topRight = [r_min, width]
                bottomLeft = [r_max, 0]
                bottomRight = [r_max, width]
                self.page_rois[i].append(np.asarray([topLeft,
                                                     topRight,
                                                     bottomRight,
                                                     bottomLeft]))

    def update_mung_alignment(self):
        """Re-computes the MuNG-notes alignment & saves the new MuNG."""

        logging.info('Updating MuNG alignment...')
        if self.page_mungos is None:
            logging.info('...no MuNG loaded!')
            return None

        aln = align_score_to_performance(self.current_score,
                                         self.current_performance,
                                         fps=FPS)
        logging.info('Total aligned pairs: {0}'.format(len(aln)))
        self.score_performance_alignment = {
            objid: note_idx
            for objid, note_idx in aln}

        self.load_performance_features()

        _perf_name = self.current_performance.name

        for i, mungos in enumerate(self.page_mungos):
            for m in mungos:
                if m.objid not in self.score_performance_alignment:
                    continue
                e_idx = self.score_performance_alignment[m.objid]
                e = self.note_events[e_idx]
                m.data['{0}_onset_seconds'.format(_perf_name)] = e[0]
                m.data['{0}_duration_seconds'.format(_perf_name)] = e[2]
                m.data['{0}_onset_frame'.format(_perf_name)] = int(self.onsets[e_idx])
                m.data['{0}_duration_frame'.format(_perf_name)] = int(self.durations[e_idx])
                m.data['{0}_note_event_idx'.format(_perf_name)] = e_idx

            page_stats = alignment_stats(mungos,
                                         self.note_events,
                                         self.score_performance_alignment)

            print('\tPage {0}: M+E {1}, M- {2}'
                  ''.format(i,
                            len(page_stats.mungos_aligned_correct_pitch),
                            len(page_stats.mungos_not_aligned_not_tied)))

        self.save_mung()

    def _load_mung_alignment(self):
        """Loads the alignment for the current performance and returns
        it as a dict: ``objid --> event_idx``."""
        aln = []
        _perf_name = self.current_performance.name
        _onset_seconds_key = '{0}_onset_seconds'.format(_perf_name)
        _onset_frames_key = '{0}_onset_frames'.format(_perf_name)
        _event_idx_key = '{0}_note_event_idx'.format(_perf_name)
        _pitch_key = 'midi_pitch_code'

        _events = self.current_performance.load_note_events()
        _events_dict = {(e[0], e[1]): i for i, e in enumerate(_events)}

        for i, mungos in enumerate(self.page_mungos):
            for m in mungos:
                if _event_idx_key not in m.data:
                    continue

                e_idx = m.data[_event_idx_key]
                aln.append((m.objid, e_idx))

        score_performance_alignment = {
            objid: note_idx
            for objid, note_idx in aln
        }
        return score_performance_alignment

    def load_mung_alignment(self):
        """Sets the current alignment (``self.score_performance_alignment``)
        using the current MuNG data. Uses the current performance."""
        aln_dict = self._load_mung_alignment()
        self.score_performance_alignment = aln_dict

    def has_alignment(self):
        aln_dict = self._load_mung_alignment()
        if len(aln_dict) == 0:
            return False
        else:
            return True

    def sort_note_coords(self):
        """ Sort note coordinates by systems (ROIs).

        By default, this filters out note coordinates that are not associated
        with any system. However, if there are no systems for a given page,
        it just returns the coords in the original order.
        """

        for page_id in range(self.n_pages):
            page_rois = self.page_rois[page_id]

            if page_rois:
                page_coords = self.page_coords[page_id]
                page_coords = sort_by_roi(page_coords, page_rois)
                # A more robust alignment procedure can be plugged here.
                self.page_coords[page_id] = page_coords

    def sort_bar_coords(self):
        """ Sort bar coords by rows """
        from sklearn.metrics.pairwise import pairwise_distances

        for page_id in range(self.n_pages):
            page_bars = self.page_bars[page_id]
            page_systems = self.page_systems[page_id]

            if page_systems.shape[0] == 0:
                continue

            # check if bars and systems are present
            if page_bars.shape[0] == 0 or page_systems.shape[0] == 0:
                break

            # compute y-coordinates
            page_systems_centers = page_systems.mean(1)[:, 0:1]
            page_bar_centers = page_bars.mean(1)[:, 0:1]

            # compute pairwise distances
            dists = pairwise_distances(page_bar_centers, page_systems_centers)

            # assign bars to systems
            bars_by_system = [np.zeros((0, 2, 2))] * page_systems.shape[0]
            for i in range(dists.shape[0]):
                min_idx = np.argmin(dists[i])
                bars = page_bars[i][np.newaxis, :, :]
                bars_by_system[min_idx] = np.vstack((bars_by_system[min_idx], bars))

            # sort bars within system
            for i, system_bars in enumerate(bars_by_system):
                sorted_idx = np.argsort(system_bars[:, 0, 1])
                bars_by_system[i] = system_bars[sorted_idx]

            self.page_bars[page_id] = np.vstack(bars_by_system)

    # -w- GUI
    def update_sheet_statistics(self):
        """ Compute sheet statistics """

        self.n_pages = len(self.sheet_pages)
        self.n_systems = np.sum([len(s) for s in self.page_systems])
        self.n_coords = np.sum([len(c) for c in self.page_coords])
        self.n_bars = np.sum([len(s) for s in self.page_bars])

        if self.gui:
            self.gui.lineEdit_nPages.setText(str(self.n_pages))
            self.gui.lineEdit_nSystems.setText(str(self.n_systems))
            self.gui.lineEdit_nCoords.setText(str(self.n_coords))
            self.gui.lineEdit_nBars.setText(str(self.n_bars))

    def save_coords(self):
        """ Save changed sheet coords """
        self.save_system_coords()
        self.save_bar_coords()
        self.save_note_coords()

        self.save_mung()

    def save_mung(self):
        """Save the current MuNG data."""
        mung_strings = {page: export_cropobject_list(self.page_mungos[page])
                        for page in range(len(self.page_mungos))}
        self.current_score.add_paged_view('mung', mung_strings,
                                          file_fmt='xml',
                                          binary=False,
                                          prefix=None,
                                          overwrite=True)

    def save_note_coords(self):
        """ Save current note coordinates. """
        coord_dir = self.current_score.coords_dir
        for i in range(len(self.page_coords)):
            coord_file = os.path.join(coord_dir, "notes_%02d.npy" % (i + 1))
            np.save(coord_file, self.page_coords[i])

    def save_bar_coords(self):
        """ Save current bar coordinates. """
        coord_dir = self.current_score.coords_dir
        for i in range(len(self.page_coords)):
            coord_file = os.path.join(coord_dir, "bars_%02d.npy" % (i + 1))
            np.save(coord_file, self.page_bars[i])

    def save_system_coords(self):
        """ Save current bar coordinates. """
        coord_dir = self.current_score.coords_dir
        for i in range(len(self.page_coords)):
            coord_file = os.path.join(coord_dir, "systems_%02d.npy" % (i + 1))
            np.save(coord_file, self.page_systems[i])

    # --x GUI
    def edit_coords(self):
        """ Edit sheet elements """

        # show sheet image along coordinates
        self.fig = plt.figure("Sheet Editor")

        # self.fig_manager = plt.get_current_fig_manager()
        # def notify_axes_change_print(fig):
        #     print('Figure {0}: Axes changed!'.format(fig))
        # self.fig.add_axobserver(notify_axes_change_print)

        # init events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.sort_note_coords()
        self.sort_bar_coords()

        self.plot_sheet()

    # r-x GUI, -wx MPL
    def plot_sheet(self, xlim=None, ylim=None):
        """
        Plot sheet image along with coordinates
        """

        if not self.gui:
            logging.info('Cannot plot sheet without GUI.')
            return
        # print('Calling plot_sheet')

        # Preserving the view/zoom history
        # prev_toolbar = plt.gcf().canvas.toolbar
        # _prev_view_elements = copy.deepcopy(prev_toolbar._views._elements)
        # _prev_position_elements = [[tuple([eee.frozen() for eee in ee]) for ee in e]
        #                    for e in prev_toolbar._positions._elements]
        # _xypress = copy.deepcopy(prev_toolbar._xypress)  # the location and axis info at the time
        #                       # of the press
        # _idPress = copy.deepcopy(prev_toolbar._idPress)
        # _idRelease = copy.deepcopy(prev_toolbar._idRelease)
        # _active = copy.deepcopy(prev_toolbar._active)
        # _lastCursor = copy.deepcopy(prev_toolbar._lastCursor)
        # _ids_zoom = copy.deepcopy(prev_toolbar._ids_zoom)
        # _zoom_mode = copy.deepcopy(prev_toolbar._zoom_mode)
        # _button_pressed = copy.deepcopy(prev_toolbar._button_pressed)  # determined by the button pressed
        #                              # at start
        # mode = copy.deepcopy(prev_toolbar.mode)  # a mode string for the status bar

        # print('...previous views: {0}'.format(_prev_view_elements))
        # print('...previous positions: {0}'.format(_prev_position_elements))
        # print('---orig positions: {0}'.format(prev_toolbar._positions._elements))

        # get data of current page
        page_id = self.gui.spinBox_page.value()

        self.fig.clf(keep_observers=True)

        # plot sheet image
        # ax = self.fig.gca()
        ax = plt.subplot(111)
        # in_ax_compr = pickle.dumps(in_ax)
        # ax_cpy = pickle.loads(in_ax_compr)
        # ax = ax_cpy.subplot(111)

        plt.subplots_adjust(top=0.98, bottom=0.05)
        plt.imshow(self.sheet_pages[page_id], cmap=plt.cm.gray, interpolation='nearest')

        plt.xlim([0, self.sheet_pages[page_id].shape[1] - 1])
        plt.ylim([self.sheet_pages[page_id].shape[0] - 1, 0])

        plt.xlabel("%d Pixel" % self.sheet_pages[page_id].shape[1], fontsize=self.axis_label_fs)
        plt.ylabel("%d Pixel" % self.sheet_pages[page_id].shape[0], fontsize=self.axis_label_fs)

        # plot note coordinates
        # They are color-coded to automatically show various situations:
        #   - cobalt: aligned to an event and they are OK
        #   - yellow: not aligned to an event, tied attribute is set
        #   - red: not aligned to an event, tied attribute is NOT set
        if self.gui.checkBox_showCoords.isChecked():
            if self._page_mungo_centroids is not None:
                mcentroids = self._page_mungo_centroids[page_id]
                # Split MuNG objects based on whether they are aligned or not
                aln_hit_m_centroids = []
                aln_miss_m_centroids = []
                aln_tied_m_centroids = []

                for mc in mcentroids:
                    mc = float(mc[0]), float(mc[1])
                    m = self._page_centroids2mungo_map[page_id][mc]
                    # Do not display system MuNGs.
                    if m.clsname == 'staff':
                        continue
                    try:
                        onset = self._mungo_onset_frame_for_current_performance(m)
                        _aln_onset, _aln_pitch = self._aligned_onset_and_pitch(m)
                    except MSMDManagerError:
                        if ('tied' in m.data) and (m.data['tied'] == 1):
                            aln_tied_m_centroids.append(mc)
                        else:
                            aln_miss_m_centroids.append(mc)
                    else:
                        aln_hit_m_centroids.append(mc)

                if len(aln_hit_m_centroids) > 0:
                    aln_hit_m_centroids = np.asarray(aln_hit_m_centroids)
                    plt.plot(aln_hit_m_centroids[:, 1], aln_hit_m_centroids[:, 0], 'co', alpha=0.4)

                if len(aln_miss_m_centroids) > 0:
                    aln_miss_m_centroids = np.asarray(aln_miss_m_centroids)
                    plt.plot(aln_miss_m_centroids[:, 1], aln_miss_m_centroids[:, 0], 'ro', alpha=0.6)

                if len(aln_tied_m_centroids) > 0:
                    aln_tied_m_centroids = np.asarray(aln_tied_m_centroids)
                    plt.plot(aln_tied_m_centroids[:, 1], aln_tied_m_centroids[:, 0], 'yo', alpha=0.4)

            else:
                plt.plot(self.page_coords[page_id][:, 1], self.page_coords[page_id][:, 0], 'co', alpha=0.6)

        # plot systems
        if self.gui.checkBox_showSystems.isChecked():
            patches = []
            for system in self.page_systems[page_id]:
                polygon = Polygon(system[:, ::-1], True)
                patches.append(polygon)

            p = PatchCollection(patches, color='k', alpha=0.2)
            ax.add_collection(p)

        # plot rois
        if self.gui.checkBox_showRois.isChecked():
            patches = []
            for roi in self.page_rois[page_id]:
                polygon = Polygon(roi[:, ::-1], True)
                patches.append(polygon)

            p = PatchCollection(patches, color='r', alpha=0.2)
            ax.add_collection(p)

        # plot bars
        if self.gui.checkBox_showBars.isChecked():
            for i, bar in enumerate(self.page_bars[page_id]):
                plt.plot(bar[:, 1], bar[:, 0], 'b-', linewidth=2, alpha=0.6)
                plt.text(bar[0, 1], bar[0, 0], i, color='b', ha='center', va='bottom',
                         bbox=dict(facecolor='w', edgecolor='b', boxstyle='round,pad=0.2'))

        # print('Requesting zoom: xlim = {0}, ylim = {1}'.format(xlim, ylim))

        # if (xlim is not None) and (ylim is not None):

        # plt.draw()

        # plt.gcf().canvas.toolbar = prev_toolbar

        # plt.xlim(xlim)
        # plt.ylim(ylim)
        #
        #
        # # In order for the Home button to work properly, we need to push the original
        # # size of the figure to the NavigationToolbar's stack of views.
        # toolbar = plt.gcf().canvas.toolbar
        # # plt.draw()
        # views = toolbar._views
        # #print(views._elements)
        # orig_xmin, orig_xmax = [0, self.sheet_pages[page_id].shape[1] - 1]
        # orig_ymin, orig_ymax = [self.sheet_pages[page_id].shape[0] - 1, 0]
        # orig_view = [(orig_xmin, orig_xmax, orig_ymin, orig_ymax)]
        # views.push(orig_view)
        # views.bubble(orig_view)
        #
        # # if len(_prev_positions) > 0:
        # #     orig_position = _prev_positions[0]
        # #     positions = toolbar._positions
        # #     positions.push(orig_position)
        # #     positions.bubble(orig_position)
        #

        plt.draw()

        plt.pause(0.1)
        # plt.show()

        plt.gcf().canvas.toolbar._update_view()

        # print('Views redrawing 2x: ')
        # print(plt.gcf().canvas.toolbar._views._elements)
        # print('Positions after drawing 2x: ')
        # print(plt.gcf().canvas.toolbar._positions._elements)

    # rwx MPL
    def on_press(self, event):
        """
        Scatter plot mouse button down event
        """
        self.press = True
        self.click_0 = [event.ydata, event.xdata]
        self.drawObjects = []

    # rwx MPL
    def on_motion(self, event):
        """
        Scatter plot mouse move event
        """

        if self.press:

            click_1 = [event.ydata, event.xdata]

            ax = plt.gca()

            while len(self.drawObjects) > 0:
                do = self.drawObjects.pop(0)
                p = do.pop(0)
                p.remove()

            do = ax.plot([self.click_0[1], self.click_0[1]], [self.click_0[0], click_1[0]],
                         'r-', linewidth=2, alpha=0.5)
            self.drawObjects.append(do)
            do = ax.plot([click_1[1], click_1[1]], [self.click_0[0], click_1[0]],
                         'r-', linewidth=2, alpha=0.5)
            self.drawObjects.append(do)
            do = ax.plot([self.click_0[1], click_1[1]], [self.click_0[0], self.click_0[0]],
                         'r-', linewidth=2, alpha=0.5)
            self.drawObjects.append(do)
            do = ax.plot([self.click_0[1], click_1[1]], [click_1[0], click_1[0]],
                         'r-', linewidth=2, alpha=0.5)
            self.drawObjects.append(do)

            plt.draw()

    # rwx MPL, r-- GUI
    def on_release(self, event):
        """
        Sheet has been clicked event
        """
        # print('Calling: on_release, with event {0}'.format(event))
        # Preserving the view/zoom history
        # _prev_views = plt.gcf().canvas.toolbar._views._elements
        # _prev_positions = plt.gcf().canvas.toolbar._positions._elements
        # print('...previous views: {0}'.format(_prev_views))
        # print('...previous positions: {0}'.format(_prev_positions))

        from sklearn.metrics.pairwise import pairwise_distances

        # reset bounding box drawing
        self.press = False
        for do in self.drawObjects:
            do.pop(0).remove()
        self.drawObjects = []

        # current page
        page_id = self.gui.spinBox_page.value()

        # position of click
        clicked = np.asarray([event.ydata, event.xdata]).reshape([1, 2])

        # right click (open spectrogram and highlight note's onset)
        # Note: this has a conflict with the usage of right-click to zoom out.
        # That was not apparent until we started playing around with retaining the zoom.
        # Maybe: better to change this to double-click?
        if event.button == 3:

            plt.figure("Spectrogram")
            plt.clf()

            # plot spectrogram
            ax1 = plt.subplot(2, 1, 1)
            plt.imshow(self.spec, aspect='auto', origin='lower', cmap=cmaps['viridis'], interpolation='nearest')

            # If possible, retrieve a MuNG note object...
            onset = None
            pitch = None
            _aln_onset = None
            _aln_pitch = None
            mung_onset_successful = True
            if self._page_mungo_centroids is not None:
                centroids = sorted(self._page_centroids2mungo_map[page_id].keys())
                dists = pairwise_distances(clicked,
                                           centroids)
                selection = np.argmin(dists)
                centroid = centroids[selection]
                mungo = self._page_centroids2mungo_map[page_id][centroid]

                if 'midi_pitch_code' in mungo.data:
                    pitch = int(mungo.data['midi_pitch_code'])

                logging.debug('Found closest mungo: {0}'.format(mungo))
                try:
                    onset = self._mungo_onset_frame_for_current_performance(mungo)
                    _aln_onset, _aln_pitch = self._aligned_onset_and_pitch(mungo)
                    plt.title("MuNG {0} (p: {1})"
                              "".format(mungo.objid,
                                        mungo.data['midi_pitch_code']))
                except MSMDManagerError:
                    mung_onset_successful = False
            else:
                mung_onset_successful = False

            # If not, fall back coords ordering.
            if not mung_onset_successful:
                logging.debug('...retrieving corresponding MuNG object not successful.')

                dists = pairwise_distances(clicked, self.page_coords[page_id])
                selection = np.argmin(dists)

                if page_id > 0:
                    offset = np.sum([len(self.page_coords[i]) for i in range(page_id)])
                    selection += offset

                onset = self.onsets[selection]

            logging.debug('Pitch: {0}, onset: {1}, aln_pitch: {2}'
                          ''.format(pitch, onset, _aln_pitch))

            plt.plot([onset, onset], [0, self.spec.shape[0]], 'w-', linewidth=2.0, alpha=0.5)

            x_min = np.max([0, onset - 100])
            x_max = x_min + 200
            plt.xlim([x_min, x_max])
            plt.ylim([0, self.spec.shape[0]])

            plt.ylabel('%d Frequency Bins' % self.spec.shape[0], fontsize=self.axis_label_fs)
            plt.xlabel('Frame', fontsize=self.axis_label_fs)

            # plot midi matrix
            if self.midi_matrix is not None:
                plt.subplot(2, 1, 2, sharex=ax1)
                plt.imshow(np.max(self.midi_matrix) - self.midi_matrix, aspect='auto', cmap=plt.cm.gray,
                           interpolation='nearest', vmin=0, vmax=np.max(self.midi_matrix))
                plt.plot([onset, onset], [0, self.midi_matrix.shape[0]], 'k-', linewidth=2.0, alpha=0.5)
                if pitch is not None:
                    plt.plot([onset], [pitch], 'ro', alpha=0.5)
                if _aln_pitch is not None:
                    plt.plot([_aln_onset], [_aln_pitch], 'bo', alpha=0.5)
                plt.ylim([0, self.midi_matrix.shape[0]])
                plt.xlim([x_min, x_max])
                plt.ylabel("%d Midi Pitches" % self.midi_matrix.shape[0], fontsize=self.axis_label_fs)
                plt.xlabel('Frame', fontsize=self.axis_label_fs)

            plt.draw()
            plt.pause(0.1)
            return

        # check for editing mode
        if not self.gui.checkBox_editSheet.isChecked() and event.button == 1:
            return

        # add system position
        if self.gui.radioButton_addSystem.isChecked():

            # compile system coordinates
            system_coords = np.zeros((4, 2))
            system_coords[0] = np.asarray([self.click_0[0], self.click_0[1]])
            system_coords[1] = np.asarray([self.click_0[0], clicked[0, 1]])
            system_coords[2] = np.asarray([clicked[0, 0], clicked[0, 1]])
            system_coords[3] = np.asarray([clicked[0, 0], self.click_0[1]])

            # find closest system
            if self.page_systems[page_id].shape[0] > 0:
                dists = pairwise_distances(system_coords[0:1, :], self.page_systems[page_id][:, 0, :])
                selection = np.argmin(dists)
            else:
                selection = -1

            self.page_systems[page_id] = np.insert(self.page_systems[page_id], selection + 1, system_coords, axis=0)
            self.click_0 = None

            self.systems_to_rois()

        # remove system position
        if self.gui.radioButton_deleteSystem.isChecked():
            import matplotlib.path as mpltPath

            # find selected system
            for i in range(len(self.page_systems[page_id])):

                path = mpltPath.Path(self.page_systems[page_id][i])
                if path.contains_point(clicked[0]):

                    # remove system
                    self.page_systems[page_id] = np.delete(self.page_systems[page_id], i, axis=0)
                    logging.info("Removed system with id:", i)
                    break

            self.systems_to_rois()

        # add bar position
        if self.gui.radioButton_addBar.isChecked():

            # get closest system
            page_systems = self.page_systems[page_id]

            # compute y-coordinates
            page_systems_centers = page_systems.mean(1)[:, 0:1]

            # compute pairwise distances
            dists = pairwise_distances(clicked[:, 0:1], page_systems_centers)

            # find closest system
            min_idx = np.argmin(dists[0])
            r0 = page_systems[min_idx, 0, 0]
            r1 = page_systems[min_idx, 3, 0]

            # compile system coordinates
            bar_coords = np.zeros((2, 2))
            bar_coords[0] = np.asarray([r0, clicked[0, 1]])
            bar_coords[1] = np.asarray([r1, clicked[0, 1]])

            # find closest system
            selection = -1

            self.page_bars[page_id] = np.insert(self.page_bars[page_id], selection + 1, bar_coords, axis=0)
            self.click_0 = None

        # remove bar position
        if self.gui.radioButton_deleteBar.isChecked():

            # find closets note
            dists = pairwise_distances(clicked, self.page_bars[page_id].mean(1))
            selection = np.argmin(dists)

            # remove coordinate
            self.page_bars[page_id] = np.delete(self.page_bars[page_id], selection, axis=0)
            logging.info("Removed bar with id:", selection)

        # add note position
        if self.gui.radioButton_addNote.isChecked():

            # find closets note
            if self.page_coords[page_id].shape[0] > 0:
                dists = pairwise_distances(clicked, self.page_coords[page_id])
                selection = np.argmin(dists)
            else:
                selection = -1

            self.page_coords[page_id] = np.insert(self.page_coords[page_id], selection + 1, clicked, axis=0)

        # remove note position
        if self.gui.radioButton_deleteNote.isChecked():

            # find closets note
            dists = pairwise_distances(clicked, self.page_coords[page_id])
            selection = np.argmin(dists)

            # remove coordinate
            self.page_coords[page_id] = np.delete(self.page_coords[page_id], selection, axis=0)
            logging.info("Removed note with id:", selection)

        # update sheet statistics
        self.update_sheet_statistics()

        # update notehead-onset alignment
        self.sort_note_coords()
        self.sort_bar_coords()

        # refresh view
        xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
        self.plot_sheet(xlim=xlim, ylim=ylim)

    # r-- GUI
    def update_staff_windows(self):
        """
        Update staff windows
        """
        self.window_top = self.gui.spinBox_window_top.value()
        self.window_bottom = self.gui.spinBox_window_bottom.value()

    def match_audio2sheet(self):
        """
        Match audio to sheet images
        """
        pass

    def init_omr(self):
        """ Initialize omr module """
        logging.info('Initializing omr ...')
        # self.status_label.setText("Initializing omr ...")

        # select model
        try:
            from omr.models import note_detector as note_model
            from omr.models import bar_detector as bar_model
            from omr.models import system_detector as system_model
            from lasagne_wrapper.network import SegmentationNetwork
            from omr.omr_app import OpticalMusicRecognizer
        except ImportError:
            logging.warn('OMR not available!')
            return

        # initialize note detection neural network
        dump_file = os.path.join(os.path.dirname(__file__),
                                 "omr_models/note_params.pkl")
        net = note_model.build_model()
        note_net = SegmentationNetwork(net, print_architecture=False)
        note_net.load(dump_file)

        # initialize bar detection neural network
        dump_file = os.path.join(os.path.dirname(__file__),
                                 "omr_models/bar_params.pkl")
        net = bar_model.build_model()
        bar_net = SegmentationNetwork(net, print_architecture=False)
        bar_net.load(dump_file)

        # initialize system detection neural network
        dump_file = os.path.join(os.path.dirname(__file__),
                                 "omr_models/system_params.pkl")
        net = system_model.build_model()
        system_net = SegmentationNetwork(net, print_architecture=False)
        system_net.load(dump_file)

        # initialize omr system
        self.omr = OpticalMusicRecognizer(note_detector=note_net,
                                          system_detector=system_net,
                                          bar_detector=bar_net)

        # self.status_label.setText("done!")

    # r-- GUI
    def detect_note_heads(self):
        """ Detect note heads in current image """
        try:
            from omr.utils.data import prepare_image
        except ImportError:
            logging.warning('OMR not available!')
            return

        logging.info('Detecting note heads ...')
        # self.status_label.setText("Detecting note heads ...")

        if self.omr is None:
            self.init_omr()

        # prepare current image for detection
        page_id = self.gui.spinBox_page.value()
        img = prepare_image(self.sheet_pages[page_id])

        # detect note heads
        self.page_coords[page_id] = self.omr.detect_notes(img)

        # refresh view
        self.sort_note_coords()

        # update sheet statistics
        self.update_sheet_statistics()

        # refresh view, if in interactive mode.
        if self.fig is not None:
            self.plot_sheet()

        # self.status_label.setText("done!")

    # r-- GUI
    def detect_bars(self):
        """ Detect bars in current image """
        try:
            from omr.utils.data import prepare_image
        except ImportError:
            logging.warn('OMR not available!')
            return

        logging.info('Detecting bars ...')
        # self.status_label.setText("Detecting bars ...")

        if self.omr is None:
            self.init_omr()

        # prepare current image for detection
        page_id = self.gui.spinBox_page.value()
        img = prepare_image(self.sheet_pages[page_id])

        # detect note heads
        self.page_bars[page_id] = self.omr.detect_bars(img, systems=self.page_systems[page_id])

        # sort detected bars
        self.sort_bar_coords()

        # update sheet statistics
        self.update_sheet_statistics()

        # refresh view, if in interactive mode.
        if self.fig is not None:
            self.plot_sheet()

        # self.status_label.setText("done!")

    # r-- GUI
    def detect_systems(self, with_omr=False):
        """ Detect system regions in current image.

        :param with_omr: If set, will use the OMR convnets. If not set,
            will use heuristics. The heuristics work on LilyPond-generated
            scores, or other scores where you can rely on stafflines being
            perfectly horizontal.
        """
        logging.info('Detecting systems ...')
        # self.status_label.setText("Detecting systems ...")

        page_id = self.gui.spinBox_page.value()
        self.detect_systems_on_page(page_id, with_omr=with_omr)

    def detect_systems_on_page(self, page_id, with_omr):
        """Detect systems on the given page."""
        if with_omr:
            if self.omr is None:
                self.init_omr()

            try:
                from omr.utils.data import prepare_image
            except ImportError:
                logging.warn('OMR not available!')
                return

            # prepare current image for detection
            img = prepare_image(self.sheet_pages[page_id])

            # detect note heads
            self.page_systems[page_id] = self.omr.detect_systems(img)

            # convert systems to rois
            self.systems_to_rois()

        else:
            # Insert the heuristic-based system region detection here
            img = self.sheet_pages[page_id]
            self.page_systems[page_id] = detect_system_regions_ly(img)
            self.systems_to_rois()

        # Propagate changes to MuNG systems!
        # Assume that we have the same number of system MuNGs as there
        # are detected system regions.
        staff_mungos = [m for m in self.page_mungos[page_id]
                        if m.clsname == 'staff']
        sorted_mungos = sorted(staff_mungos, key=lambda x: x.top)

        # self.page_systems[page_id] are the *corners*...
        sorted_regions = sorted([corners2bbox(c)
                                 for c in self.page_systems[page_id]])
        # print(sorted_regions)
        if len(staff_mungos) != len(sorted_regions):
            logging.warning('The number of systems detected by the OMR module'
                            ' ({}) does not match the number of LTR note'
                            ' groups ({}).'
                            ''.format(len(staff_mungos),
                                      len(sorted_regions)))
        for m, reg in zip(sorted_mungos, sorted_regions):
            t, l, b, r = reg
            m.x = t
            m.y = l
            m.height = b - t
            m.width = r - l
            m.to_integer_bounds()
            m.mask = np.ones((m.height, m.width), dtype='uint8')

        # Update
        self.save_mung()

        # Update coords
        self.sort_bar_coords()
        self.sort_note_coords()

        # update sheet statistics
        self.update_sheet_statistics()

        # refresh view, if in interactive mode.
        if self.fig is not None:
            self.plot_sheet()

        # self.status_label.setText("done!")

    def _mungo_onset_frame_for_current_performance(self, mungo):
        """Helper method."""
        perf_attr_string = '{0}_onset_frame' \
                           ''.format(self.current_performance.name)
        if perf_attr_string not in mungo.data:
            raise MSMDManagerError('Cannot get onset frame from MuNG'
                                   ' object {0}: data attribute {1} is'
                                   ' missing!'.format(mungo.objid,
                                                      perf_attr_string))
        return mungo.data[perf_attr_string]

    def _aligned_onset_and_pitch(self, mungo):
        """Retrieves the onset frame and pitch for the MIDI note event
        to which the given MuNG object is aligned. If the object is not
        aligned to anything, returns ``(None, None)``."""
        if self.score_performance_alignment is None:
            return None, None
        objid = mungo.objid
        if objid not in self.score_performance_alignment:
            return None, None

        event_idx = self.score_performance_alignment[objid]

        if self.note_events is None:
            return None, None
        if event_idx > self.note_events.shape[0]:
            logging.warn('Note event with number higher than no. of available'
                         ' events...? Event idx {0}, total {1}'
                         ''.format(event_idx, self.note_events.shape[0]))
            return None, None

        event = self.note_events[event_idx]
        onset = mp.notes_to_onsets([event], 1.0 / FPS)
        pitch = int(event[1])

        return onset, pitch

##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--interactive', action='store_true',
                        help='If set, will always launch interactive mode'
                             ' regardless of other arguments.')

    parser.add_argument('-d', '--data_dir', default=None,
                        help='[CLI] This is the root data directory. The piece'
                             ' dirs should be directly in this one. If running'
                             ' Sheet Manager in batch mode, this argument is'
                             ' required; if it is not given, GUI is launched.')
    parser.add_argument('-p', '--pieces', nargs='+',
                        help='[CLI] The pieces which should be processed. ')
    parser.add_argument('-a', '--all', action='store_true',
                        help='[CLI] Process all the pieces in the data_dir.'
                             ' Equivalent to -p `ls $DATA_DIR`.')
    parser.add_argument('--first_k', type=int, action='store', default=None,
                        help='[CLI] Only process the first K pieces in the data dir.'
                             ' Equivalent to -p `ls $DATA_DIR | head -n $K`.')
    parser.add_argument('--last_k', type=int, action='store', default=None,
                        help='[CLI] Only process the last K pieces in the data dir.'
                             ' Useful for 2-core parallelization on a desktop'
                             ' machine.')

    parser.add_argument('--retain_audio', action='store_true',
                        help='If set, will retain the rendered audio files,'
                             ' not just the spectrograms derived from them.'
                             ' Note that this makes the dataset *HUGE* when'
                             ' performance augmentations are done with'
                             ' multiple soundfonts and tempo settings.')
    parser.add_argument('--n_augmentation_performances', type=int, default=0,
                        help='How many performances to render beyond '
                             ' the fixed combinations.')

    parser.add_argument('-c', '--config',
                        help='Load configuration from this file. [NOT IMPLEMENTED]')
    parser.add_argument('-f', '--force',
                        help='Forces overwrite of existing data.'
                             ' [NOT IMPLEMENTED]')
    parser.add_argument('--ignore_errors', action='store_true',
                        help='If set, will not stop when the Sheet Manager'
                             ' raises an error. Instead, it will simply skip'
                             ' over the piece that raised the error.')

    parser.add_argument('--save_stats', action='store',
                        help='Pickle the alignment statistics of the pieces.')
    parser.add_argument('--stats_only', action='store_true',
                        help='If set, assumes the pieces are already processed'
                             ' and only reports the alignment stats.')
    parser.add_argument('--save_piece_lists', action='store',
                        help='Save lists of pieces to yaml file.'
                             ' (success, failed, problems)')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def launch_gui(args):
    """Launches the GUI."""
    app = QtWidgets.QApplication(sys.argv)
    myWindow = MSMDManager(interactive=True)
    if args.data_dir:
        if len(args.pieces) > 0:
            piece = args.pieces[0]
            piece_dir = os.path.join(args.data_dir, piece)
            myWindow.load_piece(piece_dir)
    myWindow.gui.show()
    app.exec_()


def run_batch_mode(args):
    """Runs MSMDManager in batch mode."""
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise OSError('Requested data dir does not exist: {0}'
                      ''.format(data_dir))
    logging.info('Processing data dir: {0}'.format(data_dir))

    pieces = args.pieces
    if args.all:
        pieces = [p for p in sorted(os.listdir(data_dir))
                  if os.path.isdir(os.path.join(data_dir, p))]
    elif args.first_k:
        pieces = [p for p in sorted(os.listdir(data_dir))
                  if os.path.isdir(os.path.join(data_dir, p))]
        pieces = pieces[:args.first_k]
    elif args.last_k:
        pieces = [p for p in sorted(os.listdir(data_dir))
                  if os.path.isdir(os.path.join(data_dir, p))]
        pieces = pieces[-args.last_k:]

    piece_dirs = []
    for p in pieces:
        piece_dir = os.path.join(data_dir, p)
        if not os.path.isdir(piece_dir):
            raise OSError('Piece does not exist: {0}'.format(piece_dir))
        piece_dirs.append(piece_dir)
    logging.info('Processing pieces:\n{0}'.format('\t'.join(pieces)))

    config_file = args.config
    if config_file and not os.path.isfile(config_file):
        raise OSError('Config file does not exist: {0}'.format(config_file))

    # We need to initialize the app to give PyQT all the context it expects
    # app = QtGui.QApplication(sys.argv, False)
    mgr = MSMDManager(interactive=False)
    # Does not do mgr.show()!
    # app.exec_()

    #####################
    # Set up the mgr. settings
    mgr.retain_audio = args.retain_audio
    mgr.n_augmentation_performances = args.n_augmentation_performances

    ##########################################################################

    piece_stats = {}
    success_pieces = []
    failed_pieces = []

    _start_time = time.clock()
    _last_time = time.clock()
    for i, (piece, piece_dir) in enumerate(zip(pieces, piece_dirs)):
        print('\n\n---------------------------------------------------'
              '\n[{0}/{1}]\tProcessing piece: {2}\n'
              ''.format(i, len(pieces), piece))

        try:
            if args.stats_only:
                page_stats, global_stats = mgr.get_stats_of_piece(piece_dir,
                                                                  require_alignment=True)
                print('Piece stats: ', len(global_stats._asdict().items()))

                # Build metadata from the stats
                print('Piece metadata: {0}'.format(mgr.piece.metadata))
                if (mgr.piece.metadata is None) or (len(mgr.piece.metadata) == 0):
                    print('\tBuilding metadata...')
                    mgr.piece.metadata['n_pages'] = mgr.n_pages
                    mgr.piece.metadata['n_performances'] = len(mgr.piece.performances)
                    mgr.piece.metadata['n_scores'] = len(mgr.piece.scores)
                    mgr.piece.metadata['aln_piece_stats'] = dict([kv for kv in global_stats._asdict().items()
                                                                  if isinstance(kv[1], int)])
                    mgr.piece.metadata['aln_page_stats'] = [dict([kv for kv in s._asdict().items()
                                                                  if isinstance(kv[1], int)])
                                                            for s in page_stats.values()]

                    # Coming up with the processing & alignment flags if the piece didn't have any.
                    if any(is_aln_problem(stats) for stats in page_stats.values()):
                        mgr.piece.metadata['aligned_well'] = False
                    elif is_aln_problem(global_stats):
                        mgr.piece.metadata['aligned_well'] = False
                    else:
                        mgr.piece.metadata['aligned_well'] = True
                        mgr.piece.metadata['processed'] = True

                    mgr.piece.dump_metadata()

                _metadata = mgr.piece.metadata

                if _metadata['processed']:
                    success_pieces.append((piece, piece_dir))
                else:
                    failed_pieces.append((piece, piece_dir))

            else:
                page_stats, global_stats = mgr.process_piece(piece_dir, workflow="ly")

                success_pieces.append((piece, piece_dir))

                # Only generate the processing success flags if the piece
                # actually has been processed.
                if mgr.piece.metadata['processed']:
                    if any(is_aln_problem(stats) for stats in page_stats.values()):
                        mgr.piece.metadata['aligned_well'] = False
                    elif is_aln_problem(global_stats):
                        mgr.piece.metadata['aligned_well'] = False
                    else:
                        mgr.piece.metadata['aligned_well'] = True
                    mgr.piece.dump_metadata()

            piece_stats[piece] = page_stats, global_stats

        except Exception as mgre:
            print('Error: {0}'.format(mgre))

            failed_pieces.append((piece, piece_dir))
            mgr.piece.metadata['processed'] = False
            mgr.piece.metadata['aligned_well'] = False
            mgr.piece.dump_metadata()

        _now = time.clock()
        print('... {0:.2f} s (Total time expired: {1:.2f} s)'
              ''.format(_now - _last_time, _now - _start_time))
        _last_time = _now

    ##########################################################################

    print('\n\nAlignment problems report:'
          '==========================\n\n')
    n_useful_notes = 0
    problem_alignment_pieces = []
    for piece in piece_stats:
        _has_problem = False
        page_stats, global_stats = piece_stats[piece]
        if is_aln_problem(global_stats):
            _has_problem = True
            print('\n{0}\n{1}\n\n'.format(piece, '-' * len(piece)))
            print('-- Global alignment problem:')
            print('\t{0} probably errors'
                  ''.format(len(global_stats.mungos_not_aligned_not_tied)))
            print('\t{0} probably correct'
                  ''.format(len(global_stats.mungos_aligned_correct_pitch)))

        problem_pages = []
        for page in page_stats:
            if is_aln_problem(page_stats[page]):
                if not _has_problem:
                    _has_problem = True
                    print('\n{0}\n{1}\n\n'.format(piece, '-' * len(piece)))
                problem_pages.append(page)

        for page in problem_pages:
            print('\n\t---- Page alignment problem: page {0}:'.format(page))
            print('\t{0} probably errors'
                  ''.format(len(page_stats[page].mungos_not_aligned_not_tied)))
            print('\t{0} probably correct'
                  ''.format(len(page_stats[page].mungos_aligned_correct_pitch)))

        if _has_problem:
            problem_alignment_pieces.append(piece)

        else:
            n_useful_notes += len(global_stats.mungos_aligned_correct_pitch)

    ##########################################################################

    print('\n\n')
    print('Pieces processed successfully: {0}'.format(len(success_pieces)))
    pprint.pprint(success_pieces)
    print('\n\n')
    print('Pieces failed: {0}'.format(len(failed_pieces)))
    pprint.pprint(failed_pieces)
    print('\n\n')
    print('Pieces processed, but with problems in alignment: {0}'
          ''.format(len(problem_alignment_pieces)))
    pprint.pprint(problem_alignment_pieces)
    print('\n\n')
    n_successfully_aligned = len(piece_stats) - len(problem_alignment_pieces)
    print('Pieces without alignment problems: {0}'
          ''.format(n_successfully_aligned))
    print('Success rate: {0:.2f}'
          ''.format(float(n_successfully_aligned) / len(pieces)))

    print('\nUseful aligned notes: {0}'.format(n_useful_notes))

    if args.save_stats:
        with open(args.save_stats, 'wb') as hdl:
            pickle.dump(piece_stats, hdl, protocol=pickle.HIGHEST_PROTOCOL)

    if args.save_piece_lists:
        pieces = dict()
        # TODO: careful this list contains the pieces with alignment problems as well
        pieces["success"] = [p[0] for p in success_pieces]
        pieces["failed"] = [p[0] for p in failed_pieces]
        pieces["problems"] = problem_alignment_pieces

        with open(args.save_piece_lists, 'wb') as hdl:
            yaml.dump(pieces, hdl, default_flow_style=False)


def _requested_interactive(args):
    if args.interactive:
        return True
    if args.data_dir is None:
        return True
    return False


def main(args):

    if _requested_interactive(args):
        launch_gui(args)

    else:
        run_batch_mode(args)

##############################################################################


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    main(args)

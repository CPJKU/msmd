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

The ``sheet_manager.py`` can be called either for batch processing as a script,
or a GUI for manually validating the annotations, especially alignment between
noteheads and onsets.
"""
from __future__ import print_function

import argparse
import logging

import time
from PyQt4 import QtCore, QtGui, Qt, uic

import os
import copy
import cv2
import glob
import shutil
import pickle
import sys
import warnings

import numpy as np

# set backend to qt
import matplotlib
from muscima.io import export_cropobject_list, parse_cropobject_list
from sheet_manager.alignments import mung_midi_from_ly_links, group_mungos_by_system, build_system_mungos_on_page, \
    align_score_to_performance
from sheet_manager.data_model.piece import Piece
from sheet_manager.data_model.util import SheetManagerDBError

matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

form_class = uic.loadUiType("gui/main.ui")[0]

from utils import sort_by_roi, natsort, get_target_shape
from pdf_parser import pdf2coords, parse_pdf
from colormaps import cmaps

from midi_parser import MidiParser, notes_to_onsets, FPS
from omr.config.settings import DATA_ROOT as ROOT_DIR
from omr.utils.data import MOZART_PIECES, BACH_PIECES, HAYDN_PIECES, BEETHOVEN_PIECES, CHOPIN_PIECES, SCHUBERT_PIECES, STRAUSS_PIECES
PIECES = MOZART_PIECES + BACH_PIECES + HAYDN_PIECES + BEETHOVEN_PIECES + CHOPIN_PIECES + SCHUBERT_PIECES + STRAUSS_PIECES

TARGET_DIR = "/home/matthias/mounts/home@rechenknecht1/Data/sheet_localization/real_music_sf"
# TARGET_DIR = "/home/matthias/cp/data/sheet_localization/real_music_sf"

#tempo_ratios = np.arange(0.9, 1.1, 0.025)  # [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
# sound_fonts = ["Acoustic_Piano", "Unison", "FluidR3_GM", "Steinway"]  # ["Acoustic_Piano", "Unison", "FluidR3_GM", "Steinway"]  # ["Steinway"]  # ["Acoustic_Piano", "Unison", "FluidR3_GM", "Steinway"]
# ["Steinway", "Acoustic_Piano", "Bright_Yamaha_Grand", "Unison", "Equinox_Grand_Pianos", "FluidR3_GM", "Premium_Grand_C7_24"]

# todo: remove this
# PIECES = BACH_PIECES + HAYDN_PIECES + BEETHOVEN_PIECES + CHOPIN_PIECES + SCHUBERT_PIECES

tempo_ratios = [0.9, 1.0, 1.1]
sound_fonts = ["FluidR3_GM"]


class SheetManagerError(Exception):
    pass


class SheetManager(QtGui.QMainWindow, form_class):
    """Processing and GUI for the Multi-Modal MIR dataset.

    The workflow is wrapped in the ``process_piece()`` method.
    """

    def __init__(self, parent=None):
        """
        Constructor
        """

        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        
        # set up status bar
        self.status_label = QtGui.QLabel("")
        self.statusbar.addWidget(self.status_label)
        
        # connect to menu
        self.actionOpen_sheet.triggered.connect(self.open_sheet)
        self.pushButton_open.clicked.connect(self.open_sheet)

        self.comboBox_score.activated.connect(self.update_current_score)
        self.comboBox_performance.activated.connect(self.update_current_performance)

        # connect to omr menu
        self.actionInit.triggered.connect(self.init_omr)
        self.actionDetect_note_heads.triggered.connect(self.detect_note_heads)
        self.actionDetect_systems.triggered.connect(self.detect_systems)
        self.actionDetect_bars.triggered.connect(self.detect_bars)

        # connect to check boxes
        self.checkBox_showCoords.stateChanged.connect(self.plot_sheet)
        self.checkBox_showRois.stateChanged.connect(self.plot_sheet)
        self.checkBox_showSystems.stateChanged.connect(self.plot_sheet)
        self.checkBox_showBars.stateChanged.connect(self.plot_sheet)

        # Workflow for generating
        self.pushButton_mxml2midi.clicked.connect(self.mxml2midi)
        self.pushButton_ly2PdfMidi.clicked.connect(self.ly2pdf_and_midi)
        self.pushButton_pdf2Img.clicked.connect(self.pdf2img)
        self.pushButton_pdf2Coords.clicked.connect(self.pdf2coords)
        self.pushButton_renderAudio.clicked.connect(self.render_audio)
        self.pushButton_extractPerformanceFeatures.clicked.connect(
            self.extract_performance_features)
        self.pushButton_audio2sheet.clicked.connect(self.match_audio2sheet)

        # Editing coords
        self.pushButton_editCoords.clicked.connect(self.edit_coords)
        self.pushButton_loadSheet.clicked.connect(self.load_sheet)
        self.pushButton_loadPerformanceFeatures.clicked.connect(
            self.load_performance_features)
        self.pushButton_loadCoords.clicked.connect(self.load_coords)
        self.pushButton_saveCoords.clicked.connect(self.save_coords)

        self.spinBox_window_top.valueChanged.connect(self.update_staff_windows)
        self.spinBox_window_bottom.valueChanged.connect(self.update_staff_windows)
        self.spinBox_page.valueChanged.connect(self.edit_coords)

        # Deprecated in favor of batch rocessing
        self.pushButton_renderAllAudios.clicked.connect(self.render_all_audios)
        self.pushButton_copySheets.clicked.connect(self.copy_sheets)
        self.pushButton_prepareAll.clicked.connect(self.prepare_all_audio)
        self.pushButton_parseAllMidis.clicked.connect(self.parse_all_midis)

        # Params for sheet editing: system bbox --> roi
        self.window_top = self.spinBox_window_top.value()
        self.window_bottom = self.spinBox_window_bottom.value()

        self.target_width = 835

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

    def open_sheet(self):
        """Choose a piece directory to open through a dialog window.
        """

        # piece root folder
        folder_name = str(QtGui.QFileDialog.getExistingDirectory(
            self,
            "Select Sheet Music",
            "."))
        self.load_piece(folder_name=folder_name)

    def process_piece(self, piece_folder, workflow="ly"):
        """Process the entire piece, using the specified workflow.

        :param piece_folder: The path of the requested piece folder.
            If the piece folder is not found, raises a ``SheetManagerError``.

        :param workflow: Which workflow to use, based on the available encoding
            of the piece. By default, uses ``"ly"``, the LilyPond workflow.
            (Currently, only the LilyPond workflow is implemented.)
        """
        if not os.path.isdir(piece_folder):
            raise SheetManagerError('Piece folder not found: {0}'
                                    ''.format(piece_folder))

        is_workflow_ly = (workflow == "ly") or (workflow == "Ly")
        if not is_workflow_ly:
            raise SheetManagerError('Only the LilyPond workflow is currently '
                                    ' supported!'
                                    ' Use arguments workflow="ly" or "Ly". '
                                    ' (Got: workflow="{0}")'.format(workflow))

        self.load_piece(piece_folder)

        if is_workflow_ly and not self.lily_file:
            raise SheetManagerError('Piece does not have LilyPond source'
                                    ' available; cannot process with workflow={0}'
                                    ''.format(workflow))

        # Load
        self.ly2pdf_and_midi()
        self.pdf2img()

        # Exploiting LilyPond point-and-click PDF annotations to get noteheads
        if is_workflow_ly:
            self.pdf2coords()

        # Audio
        self.render_audio()
        self.extract_performance_features()
        self.load_performance_features()

        # OMR
        self.load_sheet()

        if not self.omr:
            self.init_omr()
        self.detect_systems()
        self.detect_bars()

        # If we are unlucky and have no LilyPond source:
        if not is_workflow_ly:
            self.detect_note_heads()

        # Align written notes and performance onsets
        # [NOT IMPLEMENTED]
        self.sort_bar_coords()
        self.sort_note_coords()

        self.save_coords()

    def load_piece(self, folder_name):
        """Given a piece folder, set the current state of SheetManager to this piece.
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

        self._refresh_score_and_performance_selection()

        self.update_current_performance()
        self.update_current_score()

        if self.pdf_file is not None:
            self.lineEdit_pdf.setText(self.pdf_file)
        else:
            self.lineEdit_pdf.setText("")

        # Old workflow.
        self.piece_name = piece_name
        self.folder_name = piece.folder

        # compile encoding file paths
        self.mxml_file = self.piece.encodings.get('mxml', "")
        self.lily_file = self.piece.encodings.get('ly', "")
        self.midi_file = self.piece.encodings.get('midi', "")

        self.lily_normalized_file = self.piece.encodings.get('norm.ly', "")

        # update gui elements
        self.lineEdit_mxml.setText(self.mxml_file)
        self.lineEdit_lily.setText(self.lily_file)
        self.lineEdit_midi.setText(self.midi_file)

    def update_current_score(self):
        """Reacts to a change in the score that SheetManager is supposed
        to be currently processing."""
        score_name = str(self.comboBox_score.currentText())
        self.set_current_score(score_name)

    def set_current_score(self, score_name):
        """Set the current Score."""
        if (score_name is None) or (score_name == ""):
            logging.info('Selection provided no score; probably because'
                         ' no scores are available.')
            return

        try:
            current_score = self.piece.load_score(score_name)
        except SheetManagerDBError as e:
            print('Could not load score {0}: malformed?'
                  ' Error message: {1}'.format(score_name, e.message))
            return

        self.score_name = score_name
        self.current_score = current_score

        self.pdf_file = self.current_score.pdf_file
        self.sheet_folder = self.current_score.img_dir
        self.coord_folder = self.current_score.coords_dir

    def update_current_performance(self):
        perf_name = str(self.comboBox_performance.currentText())
        self.set_current_performance(perf_name)

    def set_current_performance(self, perf_name):
        if (perf_name is None) or (perf_name == ""):
            logging.info('Selection provided no performance; probably because'
                         ' no performances are available.')
            return

        try:
            current_performance = self.piece.load_performance(perf_name)
        except SheetManagerDBError as e:
            print('Could not load performance {0}: malformed?'
                  ' Error message: {1}'.format(perf_name, e.message))
            return

        self.current_performance = current_performance
        self.performance_name = perf_name

    def _refresh_score_and_performance_selection(self):
        """Synchronizes the selection of scores and performances.
        Tries to retain the previous score/performance."""
        old_score_idx = self.comboBox_score.currentIndex()
        old_score = str(self.comboBox_score.itemText(old_score_idx))

        old_perf_idx = self.comboBox_performance.currentIndex()
        old_perf = str(self.comboBox_performance.itemText(old_perf_idx))

        self.piece.update()

        self.comboBox_score.clear()
        self.comboBox_score.addItems(self.piece.available_scores)
        if old_score in self.piece.available_scores:
            idx = self.piece.available_scores.index(old_score)
            self.comboBox_score.setCurrentIndex(idx)
        else:
            self.comboBox_score.setCurrentIndex(0)
            self.update_current_score()

        self.comboBox_performance.clear()
        self.comboBox_performance.addItems(self.piece.available_performances)
        if old_perf in self.piece.available_performances:
            idx = self.piece.available_performances.index(old_perf)
            self.comboBox_performance.setCurrentIndex(idx)
        else:
            self.comboBox_performance.setCurrentIndex(0)
            self.update_current_performance()

    def ly2pdf_and_midi(self):
        """Convert the LilyPond file to PDF and MIDI (which is done automatically, if the Ly
        file contains the \midi { } directive)."""
        self.normalize_ly()

        self.status_label.setText("Convert Ly to pdf ...")
        if not os.path.isfile(self.lily_normalized_file):
            self.status_label.setText("done! (Error: LilyPond file not found!)")
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

        # Run LilyPond here.
        os.system(cmd)

        # If successful, the PDF file will be there:
        if os.path.isfile(pdf_path):
            self.piece.add_score(name=self.piece.default_score_name,
                                 pdf_file=pdf_path,
                                 overwrite=True)

            self.set_current_score(self.piece.default_score_name)
            self.lineEdit_pdf.setText(self.pdf_file)

            # Cleanup!
            os.unlink(pdf_path)

        else:
            print('Warning: LilyPond did not generate PDF file.'
                  ' Something went badly wrong.')

        # Check if the MIDI file was actually created.
        output_midi_file = os.path.join(self.folder_name, self.piece_name) + '.mid'
        if not os.path.isfile(output_midi_file):
            output_midi_file += 'i'  # If it is not *.mid, maybe it has *.midi
            if not os.path.isfile(output_midi_file):
                print('Warning: LilyPond did not generate corresponding MIDI file. Check *.ly source'
                      ' file for \\midi { } directive.')
            else:
                self.midi_file = output_midi_file
                self.lineEdit_midi.setText(self.midi_file)

        # Update with the MIDI encoding
        self.piece.update()
        self._refresh_score_and_performance_selection()

    def normalize_ly(self):
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

        if not os.path.exists(self.lily_normalized_file):
            print('Initializing normalized LilyPond file {0} failed!')
            return

        # Update to current LilyPond version
        convert_cmd = "convert-ly -e {0}".format(self.lily_normalized_file)
        print('Normalizing LilyPond: updating syntax to latest possible'
              ' version: {0}'.format(convert_cmd))
        os.system(convert_cmd)

        # Translate to default pitch language?
        translate_cmd = 'ly -i "translate english" {0}'.format(self.lily_normalized_file)
        print('Normalizing LilyPond: translating pitch names: {0}'.format(translate_cmd))
        os.system(translate_cmd)

        # Convert to absolute
        base_cmd = 'ly rel2abs'
        cmd = base_cmd + ' -i {0}'.format(self.lily_normalized_file)

        print('Normalizing LilyPond: moving to absolute: {0}'.format(cmd))
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

    def pdf2img(self):
        """ Convert pdf file to image """

        self.status_label.setText("Convert pdf to images ...")
        os.system("rm tmp/*.png")
        pdf_path = self.current_score.pdf_file
        # pdf_path = os.path.join(self.folder_name,
        #                         self.piece_name +
        #                         self.score_name + '.pdf')
        cmd = "convert -density 150 %s -quality 90 tmp/page.png" % pdf_path
        os.system(cmd)

        self.status_label.setText("Resizing images ...")
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

            if self.checkBox_showSheet.isChecked():
                plt.figure()
                plt.title('{0}, page {1}'.format(self.piece_name, i+1))
                plt.imshow(img_rsz, cmap=plt.cm.gray)

        if self.checkBox_showSheet.isChecked():
            plt.show()

        self.status_label.setText("done!")

    def pdf2coords(self):
        """Extracts notehead centroid coords and MuNG features
        from the PDF of the current Score. Saves them to the ``coords/``
        and ``mung/`` view."""
        print("Extracting coords from pdf ...")
        self.status_label.setText("Extracting coords from pdf ...")

        if not os.path.exists(self.pdf_file):
            self.status_label.setText("Extracting coords from pdf failed: PDF file not found!")
            print("Extracting coords from pdf failed: PDF file not found: {0}".format(self.pdf_file))
            return

        self.load_sheet()

        n_pages = len(self.page_coords)
        if n_pages == 0:
            self.status_label.setText("Extracting coords from pdf failed: could not find sheet!")

            print("Extracting coords from pdf failed: could not find sheet!"
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
            print("Something is wrong with the PDF vs. the generated images: page count"
                  " does not match (PDF parser: {0} pages, images: {1})."
                  " Re-generate the images from PDF.".format(len(centroids), n_pages))
            return

        # Derive no. of pages from centroids
        for page_id in centroids:
            print('Page {0}: {1} note events found'
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
        print('System objids will start from: {0}'.format(_system_start_objid))

        for i, page in enumerate(mungos.keys()):
            print('Page {0}: total MuNG objects: {1}, with pitches: {2}'
                  ''.format(page, len(mungos[page]),
                            len([m for m in mungos[page]
                                 if 'midi_pitch_code' in m.data])))
            page_system_bboxes, page_system_mungo_groups = \
                group_mungos_by_system(
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

            combined_mungos = mungos[page] + system_mungos
            mungos[page] = combined_mungos

        self.page_systems = [[] for _ in range(n_pages)]
        for page, bboxes in system_bboxes.items():
            corners = []
            print('Page {0}: system bboxes = {1}'.format(page, bboxes))
            for t, l, b, r in bboxes:
                corners.append([[t, l], [t, r], [b, r], [b, l]])
            corners_np = np.asarray(corners)
            print('Corners shape: {0}'.format(corners_np.shape))
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

        if self.checkBox_showExtractedCoords.isChecked():
            # Maybe we already have the ROIs for this image.
            self.load_coords()
            self.sort_note_coords()
            self.update_sheet_statistics()
            self.edit_coords()

        self.status_label.setText("done!")

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
        self.lineEdit_midi.setText(self.midi_file)

    def render_audio(self, performance_prefix=""):
        """
        Render audio from midi.
        """
        self.status_label.setText("Rendering audio ...")

        if 'midi' not in self.piece.encodings:
            raise SheetManagerError('Cannot render audio from current piece:'
                                    ' no MIDI encoding available!')

        from render_audio import render_audio

        if not performance_prefix:
            performance_prefix = self.piece.name

        for ratio in tempo_ratios:
            for sound_font in sound_fonts:
                performance_name = performance_prefix \
                                   + '_tempo-{0}'.format(int(1000 * ratio)) \
                                   + '_{0}'.format(sound_font)
                audio_file, perf_midi_file = render_audio(
                    self.piece.encodings['midi'],
                    sound_font=sound_font,
                    tempo_ratio=ratio, velocity=None)
                self.piece.add_performance(name=performance_name,
                                           audio_file=audio_file,
                                           midi_file=perf_midi_file,
                                           overwrite=True)
                self.piece.update()
                self._refresh_score_and_performance_selection()

                # Cleanup!
                os.unlink(audio_file)
                os.unlink(perf_midi_file)

        self.status_label.setText("done!")

    def render_all_audios(self):
        """
        Render audio from midi for all pieces.
        """
        warnings.warn('Replaced by batch processing.',
                      DeprecationWarning)
        return

        self.status_label.setText("Rendering audios ...")
        from render_audio import render_audio

        for i, piece in enumerate(PIECES):
            txt = "\033[94m" + ("\n%03d / %03d %s" % (i + 1, len(PIECES), piece)) + "\033[0m"
            print(txt)

            # compile directories
            src_dir = os.path.join(ROOT_DIR, piece)
            target_dir = os.path.join(TARGET_DIR, piece)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # clean up folder
            for f in glob.glob(os.path.join(target_dir, "audio/*")):
                os.remove(f)

            # get midi file name
            midi_file = glob.glob(os.path.join(src_dir, piece + '.mid*'))
            if len(midi_file) == 0:
                continue
            midi_file = midi_file[0]

            # render audios
            for ratio in tempo_ratios:
                for sf in sound_fonts:
                    render_audio(midi_file, sound_font=sf, tempo_ratio=ratio, velocity=None, target_dir=target_dir)

        self.status_label.setText("done!")
        print("done!")

    def extract_performance_features(self):
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
        from midi_parser import MidiParser

        self.status_label.setText("Parsing performance midi files and audios...")

        # For each performance:
        #  - load performance MIDI
        #  - compute onsets
        #  - compute MIDI matrix
        #  - save onsets as perf. feature
        #  - save MIDI matrix as perf.feature
        #  - load performance audio
        #  - compute performance spectrogram
        #  = save spectrogram as perf. feature
        for performance in self.piece.load_all_performances():
            audio_file_path = performance.audio
            midi_file_path = performance.midi
            if not os.path.isfile(midi_file_path):
                logging.warn('Performance {0} has no MIDI file, cannot'
                             ' compute onsets and MIDI matrix. Skipping.'
                             ''.format(performance.name))
                continue

            midi_parser = MidiParser(show=self.checkBox_showSpec.isChecked())
            spectrogram, onsets, midi_matrix, note_events = midi_parser.process(
                midi_file_path,
                audio_file_path,
                return_midi_matrix=True)

            performance.add_feature(spectrogram, 'spec.npy', overwrite=True)
            performance.add_feature(onsets, 'onsets.npy', overwrite=True)
            performance.add_feature(midi_matrix, 'midi.npy', overwrite=True)
            performance.add_feature(note_events, 'notes.npy', overwrite=True)

            self.onsets = onsets
            self.midi_matrix = midi_matrix
            self.spec = spectrogram
            self.note_events = note_events

        # pattern = self.folder_name + "/audio/*.mid*"
        # for midi_file_path in glob.glob(pattern):
        #     print("Processing", midi_file_path)
        #
        #     # get file names and directories
        #     directory = os.path.dirname(midi_file_path)
        #     file_name = os.path.basename(midi_file_path).split('.')[0]
        #     audio_file_path = os.path.join(directory, file_name + '.flac')
        #     spec_file_path = os.path.join(directory.replace("/audio", "/spec"), file_name + '_spec.npy')
        #     onset_file_path = os.path.join(directory.replace("/audio", "/spec"), file_name + '_onsets.npy')
        #     midi_matrix_file_path = os.path.join(directory.replace("/audio", "/spec"), file_name + '_midi.npy')
        #
        #     # check if to compute spectrogram
        #     if not self.checkBox_computeSpec.isChecked():
        #         audio_file_path = None
        #         self.spec = np.load(spec_file_path)
        #
        #     # parse midi file
        #     midi_parser = MidiParser(show=self.checkBox_showSpec.isChecked())
        #     Spec, self.onsets, self.midi_matrix = midi_parser.process(midi_file_path,
        #                                                               audio_file_path,
        #                                                               return_midi_matrix=True)
        #
        #     # save data
        #     if self.checkBox_computeSpec.isChecked():
        #         self.spec = Spec
        #         np.save(spec_file_path, self.spec)
        #         np.save(midi_matrix_file_path, self.midi_matrix)
        #
        #     np.save(onset_file_path, self.onsets)
        #

        # set number of onsets in gui
        self.lineEdit_nOnsets.setText(str(len(self.onsets)))
        
        self.status_label.setText("done!")

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
        except SheetManagerDBError as e:
            logging.warning('Loading midi matrix from current performance {0}'
                            ' failed: {1}'.format(self.current_performance.name,
                                                  e.message))
            return
        self.midi_matrix = midi_matrix

        try:
            onsets = self.current_performance.load_onsets()
        except SheetManagerDBError as e:
            logging.warning('Loading onsets from current performance {0}'
                            ' failed: {1}'.format(self.current_performance.name,
                                                  e.message))
            return
        self.onsets = onsets

        try:
            spectrogram = self.current_performance.load_spectrogram()
        except SheetManagerDBError as e:
            logging.warning('Loading spectrogram from current performance {0}'
                            ' failed: {1}'.format(self.current_performance.name,
                                                  e.message))
            return
        self.spec = spectrogram

        try:
            notes = self.current_performance.load_note_events()
        except SheetManagerDBError as e:
            logging.warning('Loading note events from current performance {0}'
                            ' failed: {1}'.format(self.current_performance.name,
                                                  e.message))
            return
        self.note_events = notes


        # set number of onsets in gui
        self.lineEdit_nOnsets.setText(str(len(self.onsets)))

    def parse_all_midis(self):
        """
        Parse midi file for all pieces.
        """
        warnings.warn('Replaced by batch processing.',
                      DeprecationWarning)
        return

        self.status_label.setText("Parsing midis ...")

        for i, piece in enumerate(PIECES):
            txt = "\033[94m" + ("\n%03d / %03d %s" % (i + 1, len(PIECES), piece)) + "\033[0m"
            print(txt)

            # create target folder
            target_dir = os.path.join(TARGET_DIR, piece)
            spec_dir = os.path.join(target_dir, "spec")
            if not os.path.exists(spec_dir):
                os.makedirs(spec_dir)

            # clean up folder
            for f in glob.glob(spec_dir + "/*"):
                os.remove(f)

            pattern = target_dir + "/audio/*.mid*"
            for midi_file_path in glob.glob(pattern):
                print("Processing", midi_file_path)

                # get file names and directories
                directory = os.path.dirname(midi_file_path)
                file_name = os.path.basename(midi_file_path).split('.')[0]
                audio_file_path = os.path.join(directory, file_name + '.flac')
                spec_file_path = os.path.join(spec_dir, file_name + '_spec.npy')
                onset_file_path = os.path.join(spec_dir, file_name + '_onsets.npy')
                midi_matrix_file_path = os.path.join(spec_dir, file_name + '_midi.npy')

                # parse midi file
                midi_parser = MidiParser(show=False)
                Spec, onsets, midi_matrix = midi_parser.process(midi_file_path, audio_file_path,
                                                                return_midi_matrix=True)

                # save data
                np.save(spec_file_path, Spec)
                np.save(onset_file_path, onsets)
                np.save(midi_matrix_file_path, midi_matrix)

                # remove audio file to save disk space
                os.remove(audio_file_path)

        self.status_label.setText("done!")
        print("done!")

    def copy_sheets(self):
        """
        Copy sheets to target folder.
        """
        warnings.warn('Copying sheets was an application-dependent operation'
                      ' for extracting aligned png/audio patches for multimodal'
                      ' score following. Replaced by batch processing.',
                      DeprecationWarning)
        return

        self.status_label.setText("Copying sheets ...")

        for i, piece in enumerate(PIECES):
            txt = "\033[94m" + ("\n%03d / %03d %s" % (i + 1, len(PIECES), piece)) + "\033[0m"
            print(txt)

            for folder in ["sheet", "coords"]:
                src_dir = os.path.join(ROOT_DIR, piece, folder)
                dst_dir = os.path.join(TARGET_DIR, piece, folder)

                if os.path.exists(dst_dir):
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)

        self.status_label.setText("done!")
        print("done!")

    def prepare_all_audio(self):
        """ Call all preparation steps for all audios """
        warnings.warn('prepare_all_audio() was an application-dependent op'
                      ' for extracting aligned png/audio patches for multimodal'
                      ' score following. Replaced by batch processing.',
                      DeprecationWarning)
        return
        # self.render_all_audios()
        # self.parse_all_midis()
        # self.copy_sheets()

    def load_sheet(self):
        """Load sheet images of current piece to prepare for OMR
        and/or coords editing.
        """
        self.status_label.setText("Loading sheet ...")

        self.sheet_pages = []
        self.page_coords = []
        self.page_rois = []
        self.page_systems = []
        self.page_bars = []

        # prepare paths
        img_files = self.current_score.image_files

        # initialize page data (currently loads just empty coords)
        n_pages = len(img_files)
        for i in xrange(n_pages):
            self.sheet_pages.append(cv2.imread(img_files[i], 0))
            
            self.page_coords.append(np.zeros((0, 2)))
            self.page_rois.append([])
            self.page_systems.append(np.zeros((0, 4, 2)))
            self.page_bars.append(np.zeros((0, 2, 2)))

        self.spinBox_page.setMaximum(n_pages - 1)

        self.update_sheet_statistics()

        # Loads the coordinates, if there are any stored
        self.load_coords()

        # Load MuNG objects, if there are any
        if 'mung' in self.current_score.views:
            self.load_mung()

        self.status_label.setText("done!")

    def load_mung(self):
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

        print('Page mungo centroids map sizes per page: {0}'
              ''.format([len(self._page_centroids2mungo_map[i])
                         for i in range(len(self.page_mungos))]))

        self.update_mung_alignment()

    def load_coords(self):
        """ Load coordinates """
        self.load_system_coords()
        self.load_bar_coords()
        self.load_note_coords()

        self.update_sheet_statistics()

    def load_system_coords(self):
        """ Load system coordinates """
        self.status_label.setText("Loading system coords ...")

        # prepare paths
        coord_dir = self.current_score.coords_dir
        coord_files = np.sort(glob.glob(coord_dir + "/systems_*.npy"))

        # load data
        if len(coord_files) > 0:
            for i in xrange(self.n_pages):
                if os.path.exists(coord_files[i]):
                    self.page_systems[i] = np.load(coord_files[i])

        # convert systems to rois
        self.systems_to_rois()

        self.status_label.setText("done!")

    def load_bar_coords(self):
        """ Load bar coordinates """
        self.status_label.setText("Loading system coords ...")

        # prepare paths
        coord_dir = self.current_score.coords_dir
        coord_files = np.sort(glob.glob(coord_dir + "/bars_*.npy"))

        # load data
        if len(coord_files) > 0:
            for i in xrange(self.n_pages):
                if os.path.exists(coord_files[i]):
                    self.page_bars[i] = np.load(coord_files[i])

        self.sort_bar_coords()

        self.status_label.setText("done!")

    def load_note_coords(self):
        """ Load note coordinates """
        self.status_label.setText("Loading coords ...")

        # prepare paths
        coord_dir = self.current_score.coords_dir
        coord_files = np.sort(glob.glob(coord_dir + "/notes_*.npy"))

        # load data
        if len(coord_files) > 0:
            for i in xrange(self.n_pages):
                if os.path.exists(coord_files[i]):
                    page_coords = np.load(coord_files[i])
                    self.page_coords[i] = page_coords

        self.sort_note_coords()

        self.status_label.setText("done!")

    def systems_to_rois(self):
        """ Convert systems to rois"""
        self.page_rois = []

        # sort systems
        for i in xrange(self.n_pages):
            sorted_idx = np.argsort(self.page_systems[i][:, 0, 0])
            self.page_systems[i] = self.page_systems[i][sorted_idx]

        for i in xrange(self.n_pages):
            width = self.sheet_pages[i].shape[1]
            self.page_rois.append([])
            for system_coords in self.page_systems[i]:
                r_min = system_coords[0, 0] - self.window_top
                r_max = system_coords[3, 0] + self.window_bottom
                topLeft = [r_min, 0]
                topRight = [r_min, width]
                bottomLeft = [r_max, 0]
                bottomRight = [r_max, width]
                self.page_rois[i].append(np.asarray([topLeft, topRight, bottomRight, bottomLeft]))

    def update_mung_alignment(self):

        print('Updating MuNG alignment...')
        if self.page_mungos is None:
            print('...no MuNG loaded!')
            return None
        aln = align_score_to_performance(self.current_score,
                                         self.current_performance)
        print('Total aligned pairs: {0}'.format(len(aln)))
        self.score_performance_alignment = {
            objid: note_idx
            for objid, note_idx in aln}

        self.load_performance_features()

        _perf_name = self.current_performance.name
        for i, mungos in enumerate(self.page_mungos):
            for m in mungos:
                if m.objid not in self.score_performance_alignment:
                    continue
                e = self.note_events[self.score_performance_alignment[m.objid]]
                onset_frame = notes_to_onsets([e], dt=1.0 / FPS)
                m.data['{0}_onset_seconds'.format(_perf_name)] = e[0]
                m.data['{0}_onset_frame'.format(_perf_name)] = int(onset_frame)

        self.save_mung()

    def sort_note_coords(self):
        """ Sort note coordinates by systems (ROIs).

        By default, this filters out note coordinates that are not associated
        with any system. However, if there are no systems for a given page,
        it just returns the coords in the original order.
        """

        for page_id in xrange(self.n_pages):
            page_rois = self.page_rois[page_id]

            if page_rois:
                page_coords = self.page_coords[page_id]
                page_coords = sort_by_roi(page_coords, page_rois)
                # A more robust alignment procedure can be plugged here.
                self.page_coords[page_id] = page_coords

    def sort_bar_coords(self):
        """ Sort bar coords by rows """
        from sklearn.metrics.pairwise import pairwise_distances

        for page_id in xrange(self.n_pages):
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
            for i in xrange(dists.shape[0]):
                min_idx = np.argmin(dists[i])
                bars = page_bars[i][np.newaxis, :, :]
                bars_by_system[min_idx] = np.vstack((bars_by_system[min_idx], bars))

            # sort bars within system
            for i, system_bars in enumerate(bars_by_system):
                sorted_idx = np.argsort(system_bars[:, 0, 1])
                bars_by_system[i] = system_bars[sorted_idx]

            self.page_bars[page_id] = np.vstack(bars_by_system)

    def update_sheet_statistics(self):
        """ Compute sheet statistics """

        self.n_pages = len(self.sheet_pages)
        self.n_systems = np.sum([len(s) for s in self.page_systems])
        self.n_coords = np.sum([len(c) for c in self.page_coords])
        self.n_bars = np.sum([len(s) for s in self.page_bars])

        self.lineEdit_nPages.setText(str(self.n_pages))
        self.lineEdit_nSystems.setText(str(self.n_systems))
        self.lineEdit_nCoords.setText(str(self.n_coords))
        self.lineEdit_nBars.setText(str(self.n_bars))
    
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
        for i in xrange(len(self.page_coords)):
            coord_file = os.path.join(coord_dir, "notes_%02d.npy" % (i + 1))
            np.save(coord_file, self.page_coords[i])

    def save_bar_coords(self):
        """ Save current bar coordinates. """
        coord_dir = self.current_score.coords_dir
        for i in xrange(len(self.page_coords)):
            coord_file = os.path.join(coord_dir, "bars_%02d.npy" % (i + 1))
            np.save(coord_file, self.page_bars[i])

    def save_system_coords(self):
        """ Save current bar coordinates. """
        coord_dir = self.current_score.coords_dir
        for i in xrange(len(self.page_coords)):
            coord_file = os.path.join(coord_dir, "systems_%02d.npy" % (i + 1))
            np.save(coord_file, self.page_systems[i])

    def edit_coords(self):
        """ Edit sheet elements """

        # show sheet image along coordinates
        self.fig = plt.figure("Sheet Editor")

        # self.fig_manager = plt.get_current_fig_manager()
        def notify_axes_change_print(fig):
            print('Figure {0}: Axes changed!'.format(fig))
        self.fig.add_axobserver(notify_axes_change_print)

        # init events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.sort_note_coords()
        self.sort_bar_coords()

        self.plot_sheet()

    def plot_sheet(self, xlim=None, ylim=None):
        """
        Plot sheet image along with coordinates
        """

        # print('Calling plot_sheet')


        # Preserving the view/zoom history
        prev_toolbar = plt.gcf().canvas.toolbar
        _prev_view_elements = copy.deepcopy(prev_toolbar._views._elements)
        _prev_position_elements = [[tuple([eee.frozen() for eee in ee]) for ee in e]
                           for e in prev_toolbar._positions._elements]
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
        page_id = self.spinBox_page.value()

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
        if self.checkBox_showCoords.isChecked():
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
                    except SheetManagerError:
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
        if self.checkBox_showSystems.isChecked():
            patches = []
            for system in self.page_systems[page_id]:
                polygon = Polygon(system[:, ::-1], True)
                patches.append(polygon)

            p = PatchCollection(patches, color='r', alpha=0.2)
            ax.add_collection(p)

        # plot rois
        if self.checkBox_showRois.isChecked():
            patches = []
            for roi in self.page_rois[page_id]:
                polygon = Polygon(roi[:, ::-1], True)
                patches.append(polygon)

            p = PatchCollection(patches, color='k', alpha=0.2)
            ax.add_collection(p)

        # plot bars
        if self.checkBox_showBars.isChecked():
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

    def on_press(self, event):
        """
        Scatter plot mouse button down event
        """
        self.press = True
        self.click_0 = [event.ydata, event.xdata]
        self.drawObjects = []

    def on_motion(self, event):
        """
        Scatter plot mouse move event
        """

        if self.press:

            click_1 = [event.ydata, event.xdata]

            ax = plt.gca()
            ax.hold(True)

            while len(self.drawObjects) > 0:
                do = self.drawObjects.pop(0)
                p = do.pop(0)
                p.remove()

            do = ax.plot([self.click_0[1], self.click_0[1]], [self.click_0[0], click_1[0]], 'r-', linewidth=2, alpha=0.5)
            self.drawObjects.append(do)
            do = ax.plot([click_1[1], click_1[1]], [self.click_0[0], click_1[0]], 'r-', linewidth=2, alpha=0.5)
            self.drawObjects.append(do)
            do = ax.plot([self.click_0[1], click_1[1]], [self.click_0[0], self.click_0[0]], 'r-', linewidth=2, alpha=0.5)
            self.drawObjects.append(do)
            do = ax.plot([self.click_0[1], click_1[1]], [click_1[0], click_1[0]], 'r-', linewidth=2, alpha=0.5)
            self.drawObjects.append(do)
            ax.hold(False)

            plt.draw()

    def on_release(self, event):
        """
        Sheet has been clicked event
        """
        print('Calling: on_release, with event {0}'.format(event))
        # Preserving the view/zoom history
        _prev_views = plt.gcf().canvas.toolbar._views._elements
        _prev_positions = plt.gcf().canvas.toolbar._positions._elements
        print('...previous views: {0}'.format(_prev_views))
        print('...previous positions: {0}'.format(_prev_positions))

        from sklearn.metrics.pairwise import pairwise_distances

        # reset bounding box drawing
        self.press = False
        for do in self.drawObjects:
            do.pop(0).remove()
        self.drawObjects = []

        # current page
        page_id = self.spinBox_page.value()
        
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
            plt.subplot(2, 1, 1)
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

                print('Found closest mungo: {0}'.format(mungo))
                try:
                    onset = self._mungo_onset_frame_for_current_performance(mungo)
                    _aln_onset, _aln_pitch = self._aligned_onset_and_pitch(mungo)
                    plt.title("MuNG {0} (p: {1})"
                              "".format(mungo.objid,
                                        mungo.data['midi_pitch_code']))
                except SheetManagerError:
                    mung_onset_successful = False
            else:
                mung_onset_successful = False

            # If not, fall back coords ordering.
            if not mung_onset_successful:
                print('...retrieving corresponding MuNG object not successful.')

                dists = pairwise_distances(clicked, self.page_coords[page_id])
                selection = np.argmin(dists)

                if page_id > 0:
                    offset = np.sum([len(self.page_coords[i]) for i in xrange(page_id)])
                    selection += offset
            
                onset = self.onsets[selection]

            print('Pitch: {0}, onset: {1}, aln_pitch: {2}'.format(pitch, onset, _aln_pitch))

            plt.plot([onset, onset], [0, self.spec.shape[0]], 'w-', linewidth=2.0, alpha=0.5)

            x_min = np.max([0, onset - 100])
            x_max = x_min + 200
            plt.xlim([x_min, x_max])
            plt.ylim([0, self.spec.shape[0]])

            plt.ylabel('%d Frequency Bins' % self.spec.shape[0], fontsize=self.axis_label_fs)
            plt.xlabel('Frame', fontsize=self.axis_label_fs)

            # plot midi matrix
            if self.midi_matrix is not None:
                plt.subplot(2, 1, 2)
                plt.imshow(np.max(self.midi_matrix) - self.midi_matrix, aspect='auto', cmap=plt.cm.gray, interpolation='nearest',
                           vmin=0, vmax=np.max(self.midi_matrix))
                plt.plot([onset, onset], [0, self.midi_matrix.shape[0]], 'k-', linewidth=2.0, alpha=0.5)
                if pitch is not None:
                    plt.plot([onset], [pitch], 'ro', alpha=0.5)
                if _aln_pitch is not None:
                    plt.plot([_aln_onset], [_aln_pitch], 'bo', alpha=0.5)
                plt.ylim([0, self.midi_matrix.shape[0]])
                plt.xlim([x_min, x_max])
                plt.ylabel("%d Midi Pitches" % self.midi_matrix.shape[0])
                plt.xlabel('Frame', fontsize=self.axis_label_fs)

            plt.draw()
            plt.pause(0.1)
            return

        # check for editing mode
        if not self.checkBox_editSheet.isChecked() and event.button == 1:
            return

        # add system position
        if self.radioButton_addSystem.isChecked():

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
        if self.radioButton_deleteSystem.isChecked():
            import matplotlib.path as mpltPath

            # find selected system
            for i in xrange(len(self.page_systems[page_id])):

                path = mpltPath.Path(self.page_systems[page_id][i])
                if path.contains_point(clicked[0]):

                    # remove system
                    self.page_systems[page_id] = np.delete(self.page_systems[page_id], i, axis=0)
                    print("Removed system with id:", i)
                    break

            self.systems_to_rois()

        # add bar position
        if self.radioButton_addBar.isChecked():

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
        if self.radioButton_deleteBar.isChecked():

            # find closets note
            dists = pairwise_distances(clicked, self.page_bars[page_id].mean(1))
            selection = np.argmin(dists)

            # remove coordinate
            self.page_bars[page_id] = np.delete(self.page_bars[page_id], selection, axis=0)
            print("Removed bar with id:", selection)

        # add note position
        if self.radioButton_addNote.isChecked():

            # find closets note
            if self.page_coords[page_id].shape[0] > 0:
                dists = pairwise_distances(clicked, self.page_coords[page_id])
                selection = np.argmin(dists)
            else:
                selection = -1

            self.page_coords[page_id] = np.insert(self.page_coords[page_id], selection + 1, clicked, axis=0)

        # remove note position
        if self.radioButton_deleteNote.isChecked():
            
            # find closets note
            dists = pairwise_distances(clicked, self.page_coords[page_id])
            selection = np.argmin(dists)
            
            # remove coordinate
            self.page_coords[page_id] = np.delete(self.page_coords[page_id], selection, axis=0)
            print("Removed note with id:", selection)
        
        # update sheet statistics
        self.update_sheet_statistics()
        
        # update notehead-onset alignment
        self.sort_note_coords()
        self.sort_bar_coords()

        # refresh view
        xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
        self.plot_sheet(xlim=xlim, ylim=ylim)

    def update_staff_windows(self):
        """
        Update staff windows
        """
        self.window_top = self.spinBox_window_top.value()
        self.window_bottom = self.spinBox_window_bottom.value()
    
    def match_audio2sheet(self):
        """
        Match audio to sheet images
        """
        pass

    def init_omr(self):
        """ Initialize omr module """
        print('Initializing omr ...')
        self.status_label.setText("Initializing omr ...")

        # select model
        from omr.models import note_detector as note_model
        from omr.models import bar_detector as bar_model
        from omr.models import system_detector as system_model
        from lasagne_wrapper.network import SegmentationNetwork
        from omr.omr_app import OpticalMusicRecognizer

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

        self.status_label.setText("done!")

    def detect_note_heads(self):
        """ Detect note heads in current image """
        from omr.utils.data import prepare_image

        print('Detecting note heads ...')
        self.status_label.setText("Detecting note heads ...")

        if self.omr is None:
            self.init_omr()

        # prepare current image for detection
        page_id = self.spinBox_page.value()
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

        self.status_label.setText("done!")

    def detect_bars(self):
        """ Detect bars in current image """
        from omr.utils.data import prepare_image

        print('Detecting bars ...')
        self.status_label.setText("Detecting bars ...")

        if self.omr is None:
            self.init_omr()

        # prepare current image for detection
        page_id = self.spinBox_page.value()
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

        self.status_label.setText("done!")

    def detect_systems(self):
        """ Detect systems in current image """
        from omr.utils.data import prepare_image

        print('Detecting systems ...')
        self.status_label.setText("Detecting systems ...")

        if self.omr is None:
            self.init_omr()

        # prepare current image for detection
        page_id = self.spinBox_page.value()
        img = prepare_image(self.sheet_pages[page_id])

        # detect note heads
        self.page_systems[page_id] = self.omr.detect_systems(img)

        # convert systems to rois
        self.systems_to_rois()

        # Update coords
        self.sort_bar_coords()
        self.sort_note_coords()

        # update sheet statistics
        self.update_sheet_statistics()

        # refresh view, if in interactive mode.
        if self.fig is not None:
            self.plot_sheet()

        self.status_label.setText("done!")

    def _mungo_onset_frame_for_current_performance(self, mungo):
        """Helper method."""
        perf_attr_string = '{0}_onset_frame' \
                           ''.format(self.current_performance.name)
        if perf_attr_string not in mungo.data:
            raise SheetManagerError('Cannot get onset frame from MuNG'
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
        onset = notes_to_onsets([event], 1.0 / FPS)
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
    parser.add_argument('-c', '--config',
                        help='Load configuration from this file.')
    parser.add_argument('-f', '--force',
                        help='Forces overwrite of existing data.'
                             ' [NOT IMPLEMENTED]')
    parser.add_argument('--ignore_errors', action='store_true',
                        help='If set, will not stop when the Sheet Manager'
                             ' raises an error. Instead, it will simply skip'
                             ' over the piece that raised the error.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def launch_gui(args):
    """Launches the GUI."""
    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    app = QtGui.QApplication(sys.argv)
    myWindow = SheetManager()
    if args.data_dir:
        if len(args.pieces) > 0:
            piece = args.pieces[0]
            piece_dir = os.path.join(args.data_dir, piece)
            myWindow.load_piece(piece_dir)
    myWindow.show()
    app.exec_()


def run_batch_mode(args):
    """Runs SheetManager in batch mode."""
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise OSError('Requested data dir does not exist: {0}'
                      ''.format(data_dir))
    logging.info('Processing data dir: {0}'.format(data_dir))

    pieces = args.pieces
    if args.all:
        pieces = os.listdir(data_dir)

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
    app = QtGui.QApplication(sys.argv)
    mgr = SheetManager()
    # Does not do mgr.show()!
    # app.exec_()

    _start_time = time.clock()
    _last_time = time.clock()
    for i, (piece, piece_dir) in enumerate(zip(pieces, piece_dirs)):
        print('[{0}/{1}]\tProcessing piece: {2}'.format(i, len(pieces), piece))

        try:
            mgr.process_piece(piece_dir, workflow="ly")
        except SheetManagerError as mgre:
            print('SheetManagerError: {0}'.format(mgre.message))

        _now = time.clock()
        print('... {0:.2f} s (Total time expired: {1:.2f} s)'
              ''.format(_now - _last_time, _now - _start_time))
        _last_time = _now

    print('Processing finished!')


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

    main(args)

from __future__ import print_function
from PyQt4 import QtCore, QtGui, Qt, uic

import os
import copy
import cv2
import glob
import shutil
import pickle
import numpy as np

# set backend to qt
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

form_class = uic.loadUiType("gui/main.ui")[0]

from utils import sort_by_roi, natsort, get_target_shape
from pdf_parser import pdf2coords
from colormaps import cmaps

from midi_parser import MidiParser

from omr.config.settings import DATA_ROOT as ROOT_DIR
from omr.utils.data import MOZART_PIECES, BACH_PIECES, HAYDN_PIECES, BEETHOVEN_PIECES, CHOPIN_PIECES, SCHUBERT_PIECES, STRAUSS_PIECES
PIECES = MOZART_PIECES + BACH_PIECES + HAYDN_PIECES + BEETHOVEN_PIECES + CHOPIN_PIECES + SCHUBERT_PIECES + STRAUSS_PIECES

TARGET_DIR = "/home/matthias/mounts/home@rechenknecht1/Data/sheet_localization/real_music_sf"
# TARGET_DIR = "/home/matthias/cp/data/sheet_localization/real_music_sf"

tempo_ratios = np.arange(0.9, 1.1, 0.025)  # [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
sound_fonts = ["Acoustic_Piano", "Unison", "FluidR3_GM", "Steinway"]  # ["Acoustic_Piano", "Unison", "FluidR3_GM", "Steinway"]  # ["Steinway"]  # ["Acoustic_Piano", "Unison", "FluidR3_GM", "Steinway"]
# ["Steinway", "Acoustic_Piano", "Bright_Yamaha_Grand", "Unison", "Equinox_Grand_Pianos", "FluidR3_GM", "Premium_Grand_C7_24"]

# todo: remove this
PIECES = BACH_PIECES + HAYDN_PIECES + BEETHOVEN_PIECES + CHOPIN_PIECES + SCHUBERT_PIECES
tempo_ratios = [1.0]
sound_fonts = ["Steinway"]


class SheetManager(QtGui.QMainWindow, form_class):
    """
    Gui for managing annotated sheet music
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

        # connect to buttons
        self.pushButton_mxml2midi.clicked.connect(self.mxml2midi)
        self.pushButton_ly2PdfMidi.clicked.connect(self.ly2pdf_and_midi)
        self.pushButton_pdf2Coords.clicked.connect(self.pdf2coords)
        self.pushButton_loadSpectrogram.clicked.connect(self.load_spectrogram)
        self.pushButton_renderAudio.clicked.connect(self.render_audio)
        self.pushButton_renderAllAudios.clicked.connect(self.render_all_audios)
        self.pushButton_parseMidi.clicked.connect(self.parse_midi)
        self.pushButton_parseAllMidis.clicked.connect(self.parse_all_midis)
        self.pushButton_copySheets.clicked.connect(self.copy_sheets)
        self.pushButton_prepareAll.clicked.connect(self.prepare_all)
        self.pushButton_editCoords.clicked.connect(self.edit_coords)
        self.pushButton_loadSheet.clicked.connect(self.load_sheet)
        self.pushButton_loadCoords.clicked.connect(self.load_coords)
        self.pushButton_saveCoords.clicked.connect(self.save_coords)
        self.pushButton_audio2sheet.clicked.connect(self.match_audio2sheet)
        self.pushButton_pdf2Img.clicked.connect(self.pdf2img)

        self.spinBox_window_top.valueChanged.connect(self.update_staff_windows)
        self.spinBox_window_bottom.valueChanged.connect(self.update_staff_windows)
        self.spinBox_page.valueChanged.connect(self.edit_coords)

        self.window_top = self.spinBox_window_top.value()
        self.window_bottom = self.spinBox_window_bottom.value()

        self.target_width = 835

        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        self.lily_file = None
        self.mxml_file = None
        self.pdf_file = None
        self.midi_file = None

        self.fig = None
        self.fig_manager = None
        self.click_0 = None
        self.click_1 = None
        self.press = False
        self.drawObjects = []

        self.sheet_pages = None
        self.page_coords = None
        self.page_systems = None
        self.page_bars = None

        self.sheet_version = None
        self.sheet_folder = None
        self.coord_folder = None

        self.folder_name = None
        self.piece_name = None

        self.omr = None

        self.axis_label_fs = 16

        self.midi_matrix = None

    def open_sheet(self):
        """
        Open entire folder
        """

        # piece root folder
        self.folder_name = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Sheet Music", "."))

        # name of piece
        self.piece_name = os.path.basename(self.folder_name)

        # selected sheet version
        self.sheet_version = self.spinBox_sheetVersion.value()
        self.sheet_version = "_%02d" % self.sheet_version if self.sheet_version > 0 else ""

        # initialize folder structure
        self.sheet_folder = "sheet" + self.sheet_version
        self.coord_folder = "coords" + self.sheet_version
        for sub_dir in [self.sheet_folder, "spec", "audio", self.coord_folder]:
            sub_path = os.path.join(self.folder_name, sub_dir)
            if not os.path.exists(sub_path):
                os.mkdir(sub_path)

        # compile file paths
        self.mxml_file = os.path.join(self.folder_name, self.piece_name + '.xml')
        self.lily_file = os.path.join(self.folder_name, self.piece_name + '.ly')
        self.midi_file = glob.glob(os.path.join(self.folder_name, self.piece_name) + '.mid*')
        self.pdf_file = os.path.join(self.folder_name, self.piece_name + self.sheet_version + '.pdf')

        if len(self.midi_file) == 0:
            self.midi_file = ""
        else:
            self.midi_file = self.midi_file[0]

        # check if files exist
        if not os.path.exists(self.mxml_file):
            self.mxml_file = ""

        if not os.path.exists(self.lily_file):
            self.lily_file = ""

        if not os.path.exists(self.pdf_file):
            self.pdf_file = ""

        # set gui elements
        self.lineEdit_mxml.setText(self.mxml_file)
        self.lineEdit_lily.setText(self.lily_file)
        self.lineEdit_midi.setText(self.midi_file)
        self.lineEdit_pdf.setText(self.pdf_file)

    def ly2pdf_and_midi(self):
        """Convert the LilyPond file to PDF and MIDI (which is done automatically, if the Ly
        file contains the \midi { } directive)."""
        self.status_label.setText("Convert Ly to pdf ...")
        if not os.path.isfile(self.lily_file):
            self.status_label.setText("done! (Error: LilyPond file not found!)")
            print('Error: LilyPond file not found!')
            return

        # Set PDF paths. LilyPond needs the output path without the .pdf suffix
        pdf_path_nosuffix = os.path.join(self.folder_name, self.piece_name + self.sheet_version)
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

        cmd = cmd_base + ' -o {0} '.format(pdf_path_nosuffix) + cmd_options + ' {0}'.format(self.lily_file)

        # Run LilyPond here.
        os.system(cmd)

        # If successful, the PDF file will be there:
        if os.path.isfile(pdf_path):
            self.pdf_file = pdf_path
            self.lineEdit_pdf.setText(self.pdf_file)
        else:
            print('Warning: LilyPond did not generate PDF file.')
            return

        # Check if the MIDI file was actually created.
        output_midi_file = os.path.join(self.folder_name, self.piece_name) + '.mid'
        if not os.path.isfile(output_midi_file):
            output_midi_file += 'i'
            if not os.path.isfile(output_midi_file):
                print('Warning: LilyPond did not generate corresponding MIDI file. Check *.ly source'
                      ' file for \\midi { } directive.')
                return
        self.midi_file = output_midi_file
        self.lineEdit_midi.setText(self.midi_file)

    def pdf2img(self):
        """ Convert pdf file to image """

        self.status_label.setText("Convert pdf to images ...")
        os.system("rm tmp/*.png")
        pdf_path = os.path.join(self.folder_name, self.piece_name + self.sheet_version + '.pdf')
        cmd = "convert -density 150 %s -quality 90 tmp/page.png" % pdf_path
        os.system(cmd)

        self.status_label.setText("Resizing images ...")
        img_dir = os.path.join(self.folder_name, self.sheet_folder)

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

        self.status_label.setText("done!")

    def pdf2coords(self):
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
        centroids = pdf2coords(self.pdf_file, target_width=self.target_width)

        # Check that the PDF corresponds to the generated images...
        if len(centroids) != n_pages:
            print("Something is wrong with the PDF vs. the generated images: page count"
                  " does not match (PDF parser: {0} pages, images: {1})."
                  " Re-generate the images from PDF.".format(len(centroids), n_pages))
            return

        # Derive no. of pages from centroids
        for page_id in centroids:
            print('Page {0}: {1} note events found'.format(page_id, centroids[page_id].shape[0]))
            self.page_coords[page_id] = centroids[page_id]

        # Save the coordinates
        self.save_coords()

        # refresh view
        self.sort_note_coords()

        # update sheet statistics
        self.update_sheet_statistics()

        # self.plot_sheet()

        self.status_label.setText("done!")

    def mxml2midi(self):
        """
        Convert mxml to midi file
        """

        # generate midi
        os.system("musicxml2ly --midi -a %s -o tmp/tmp.ly" % self.mxml_file)
        os.system("lilypond -o tmp/tmp tmp/tmp.ly")

        # copy midi file
        self.midi_file = os.path.join(self.folder_name, self.piece_name + '.midi')
        os.system("cp tmp/tmp.midi %s" % self.midi_file)
        self.lineEdit_midi.setText(self.midi_file)

    def render_audio(self):
        """
        Render audio from midi
        """
        self.status_label.setText("Rendering audio ...")
        from render_audio import render_audio
        for ratio in tempo_ratios:
            render_audio(self.midi_file, sound_font="FluidR3_GM", tempo_ratio=ratio, velocity=None)
        self.status_label.setText("done!")

    def render_all_audios(self):
        """
        Render audio from midi
        """
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

    def parse_midi(self):
        """
        Parse midi file
        """
        from midi_parser import MidiParser

        self.status_label.setText("Parsing midi ...")
        
        pattern = self.folder_name + "/audio/*.mid*"
        for midi_file_path in glob.glob(pattern):
            print("Processing", midi_file_path)

            # get file names and directories
            directory = os.path.dirname(midi_file_path)
            file_name = os.path.basename(midi_file_path).split('.')[0]
            audio_file_path = os.path.join(directory, file_name + '.flac')
            spec_file_path = os.path.join(directory.replace("/audio", "/spec"), file_name + '_spec.npy')
            onset_file_path = os.path.join(directory.replace("/audio", "/spec"), file_name + '_onsets.npy')
            midi_matrix_file_path = os.path.join(directory.replace("/audio", "/spec"), file_name + '_midi.npy')

            # check if to compute spectrogram
            if not self.checkBox_computeSpec.isChecked():
                audio_file_path = None
                self.spec = np.load(spec_file_path)
            
            # parse midi file
            midi_parser = MidiParser(show=self.checkBox_showSpec.isChecked())
            Spec, self.onsets, self.midi_matrix = midi_parser.process(midi_file_path, audio_file_path,
                                                                      return_midi_matrix=True)

            # save data
            if self.checkBox_computeSpec.isChecked():
                self.spec = Spec
                np.save(spec_file_path, self.spec)
                np.save(midi_matrix_file_path, self.midi_matrix)

            np.save(onset_file_path, self.onsets)
        
        # set number of onsets in gui
        self.lineEdit_nOnsets.setText(str(len(self.onsets)))
        
        self.status_label.setText("done!")

    def load_spectrogram(self):
        """
        Load spectrogram and onsets
        """

        # set file paths
        pattern = self.folder_name + "/audio/*.mid*"
        midi_file_path = glob.glob(pattern)[0]
        directory = os.path.dirname(midi_file_path)
        file_name = os.path.basename(midi_file_path).split('.')[0]
        spec_file_path = os.path.join(directory.replace("/audio", "/spec"), file_name + '_spec.npy')
        onset_file_path = os.path.join(directory.replace("/audio", "/spec"), file_name + '_onsets.npy')

        print("Loading spectrogram from:")
        print(spec_file_path)
        self.spec = np.load(spec_file_path)
        self.onsets = np.load(onset_file_path)

        # set number of onsets in gui
        self.lineEdit_nOnsets.setText(str(len(self.onsets)))

    def parse_all_midis(self):
        """
        Parse midi file
        """

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
        Copy sheets to target folder
        """
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

    def prepare_all(self):
        """ Call all preparation steps for all audios """
        self.render_all_audios()
        self.parse_all_midis()
        self.copy_sheets()

    def load_sheet(self):
        """
        Load sheet image
        """
        self.status_label.setText("Loading sheet ...")

        self.sheet_pages = []
        self.page_coords = []
        self.page_rois = []
        self.page_systems = []
        self.page_bars = []

        # prepare paths
        sheet_dir = os.path.join(self.folder_name, self.sheet_folder)
        img_files = np.sort(glob.glob(sheet_dir + "/*.*"))

        # load data
        n_pages = len(img_files)
        for i in xrange(n_pages):
            self.sheet_pages.append(cv2.imread(img_files[i], 0))
            
            self.page_coords.append(np.zeros((0, 2)))
            self.page_rois.append([])
            self.page_systems.append(np.zeros((0, 4, 2)))
            self.page_bars.append(np.zeros((0, 2, 2)))

        self.spinBox_page.setMaximum(n_pages - 1)
        
        self.update_sheet_statistics()
        
        self.status_label.setText("done!")

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
        coord_dir = os.path.join(self.folder_name, self.coord_folder)
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
        coord_dir = os.path.join(self.folder_name, self.coord_folder)
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
        coord_dir = os.path.join(self.folder_name, self.coord_folder)
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
        coord_dir = os.path.join(self.folder_name, self.coord_folder)
        for i in xrange(len(self.page_coords)):
            coord_file = os.path.join(coord_dir, "notes_%02d.npy" % (i + 1))
            np.save(coord_file, self.page_coords[i])

            coord_file = os.path.join(coord_dir, "bars_%02d.npy" % (i + 1))
            np.save(coord_file, self.page_bars[i])

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
        if self.checkBox_showCoords.isChecked():
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

        print('Requesting zoom: xlim = {0}, ylim = {1}'.format(xlim, ylim))

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

        # toolbar._views._elements += _prev_views  # Contains the current view as well
        # toolbar._positions._elements += _prev_positions
        #
        # # else:
        # #     plt.xlim([0, self.sheet_pages[page_id].shape[1] - 1])
        # #     plt.ylim([self.sheet_pages[page_id].shape[0] - 1, 0])
        # in_ax.upate_from(ax)
        ### toolbar = plt.gcf().canvas.toolbar
        ### toolbar._views._elements = _prev_view_elements
        ### toolbar._positions._elements = _prev_position_elements
        # toolbar._xypress = copy.deepcopy(prev_toolbar._xypress)  # the location and axis info at the time
        #                       # of the press
        # toolbar._idPress = copy.deepcopy(prev_toolbar._idPress)
        # toolbar._idRelease = copy.deepcopy(prev_toolbar._idRelease)
        # toolbar._active = copy.deepcopy(prev_toolbar._active)
        # toolbar._lastCursor = copy.deepcopy(prev_toolbar._lastCursor)
        # toolbar._ids_zoom = copy.deepcopy(prev_toolbar._ids_zoom)
        # toolbar._zoom_mode = copy.deepcopy(prev_toolbar._zoom_mode)
        # toolbar._button_pressed = copy.deepcopy(prev_toolbar._button_pressed)  # determined by the button pressed
        #                              # at start
        # toolbar.mode = copy.deepcopy(prev_toolbar.mode)  # a mode string for the status bar

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

        #

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
            
            dists = pairwise_distances(clicked, self.page_coords[page_id])
            selection = np.argmin(dists)
            
            if page_id > 0:
                offset = np.sum([len(self.page_coords[i]) for i in xrange(page_id)])
                selection += offset
            
            onset = self.onsets[selection]
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

        self.status_label.setText("Initializing omr ...")

        # select model
        from omr.models import note_detector as note_model
        from omr.models import bar_detector as bar_model
        from omr.models import system_detector as system_model
        from lasagne_wrapper.network import SegmentationNetwork
        from omr.omr_app import OpticalMusicRecognizer

        # initialize note detection neural network
        dump_file = "omr_models/note_params.pkl"
        net = note_model.build_model()
        note_net = SegmentationNetwork(net, print_architecture=False)
        note_net.load(dump_file)

        # initialize bar detection neural network
        dump_file = "omr_models/bar_params.pkl"
        net = bar_model.build_model()
        bar_net = SegmentationNetwork(net, print_architecture=False)
        bar_net.load(dump_file)

        # initialize system detection neural network
        dump_file = "omr_models/system_params.pkl"
        net = system_model.build_model()
        system_net = SegmentationNetwork(net, print_architecture=False)
        system_net.load(dump_file)

        # initialize omr system
        self.omr = OpticalMusicRecognizer(note_detector=note_net, system_detector=system_net, bar_detector=bar_net)

        self.status_label.setText("done!")

    def detect_note_heads(self):
        """ Detect note heads in current image """
        from omr.utils.data import prepare_image

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

        self.plot_sheet()

        self.status_label.setText("done!")

    def detect_bars(self):
        """ Detect bars in current image """
        from omr.utils.data import prepare_image

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

        # refresh view
        self.plot_sheet()

        self.status_label.setText("done!")

    def detect_systems(self):
        """ Detect systems in current image """
        from omr.utils.data import prepare_image

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

        # update sheet statistics
        self.update_sheet_statistics()

        # refresh view
        self.plot_sheet()

        self.status_label.setText("done!")

if __name__ == "__main__":
    """ main """
    import sys
    app = QtGui.QApplication(sys.argv)
    myWindow = SheetManager()
    myWindow.show()
    app.exec_()

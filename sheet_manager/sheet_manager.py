
from PyQt4 import QtCore, QtGui, Qt, uic

import os
import cv2
import glob
import pickle
import numpy as np

# set backend to qt
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

form_class = uic.loadUiType("gui/main.ui")[0]

from utils import sort_by_roi, natsort


BPMs = [120]


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

        # connect to buttons
        self.pushButton_mxml2midi.clicked.connect(self.mxml2midi)
        self.pushButton_renderAudio.clicked.connect(self.render_audio)
        self.pushButton_parseMidi.clicked.connect(self.parse_midi)
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

        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        self.fig = None
        self.click_0 = None
        self.click_1 = None
        self.press = False
        self.drawObjects = []

        self.sheet_pages = None
        self.page_coords = None
        self.page_systems = None

        self.folder_name = None
        self.piece_name = None

        self.omr = None

    def open_sheet(self):
        """
        Open entire folder
        """

        # piece root folder
        self.folder_name = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Sheet Music", "."))

        # initialize folder structure
        for sub_dir in ["sheet", "spec", "audio", "coords"]:
            sub_path = os.path.join(self.folder_name, sub_dir)
            if not os.path.exists(sub_path):
                os.mkdir(sub_path)

        # name of piece
        self.piece_name = os.path.basename(self.folder_name)

        # compile file paths
        self.mxml_file = os.path.join(self.folder_name, self.piece_name + '.xml')
        self.lily_file = os.path.join(self.folder_name, self.piece_name + '.ly')
        self.midi_file = os.path.join(self.folder_name, self.piece_name + '.midi')

        # check if files exist
        if not os.path.exists(self.mxml_file):
            self.mxml_file = ""

        if not os.path.exists(self.lily_file):
            self.lily_file = ""

        if not os.path.exists(self.midi_file):
            self.midi_file = ""

        # set gui elements
        self.lineEdit_mxml.setText(self.mxml_file)
        self.lineEdit_lily.setText(self.lily_file)
        self.lineEdit_midi.setText(self.midi_file)

    def pdf2img(self):
        """ Convert pdf file to image """

        self.status_label.setText("Convert pdf to images ...")
        os.system("rm tmp/*.png")
        pdf_path = os.path.join(self.folder_name, self.piece_name + '.pdf')
        cmd = "convert -density 150 %s -quality 90 tmp/page.png" % pdf_path
        os.system(cmd)

        self.status_label.setText("Resizing images ...")
        img_dir = os.path.join(self.folder_name, "sheet")
        target_width = 835

        file_paths = glob.glob("tmp/*.png")
        file_paths = natsort(file_paths)
        for i, img_path in enumerate(file_paths):
            img = cv2.imread(img_path, -1)

            # compute resize stats
            ratio = float(target_width) / img.shape[1]
            target_height = img.shape[0] * ratio
            target_width = int(target_width)
            target_height = int(target_height)

            img_rsz = cv2.resize(img, (target_width, target_height))

            out_path = os.path.join(img_dir, "%02d.png" % (i + 1))
            cv2.imwrite(out_path, img_rsz)

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
        from score_alignment.lilypond_note_coords.render_audio import render_audio
        for bpm in BPMs:
            render_audio(self.midi_file, sound_font="Steinway", bpm=bpm, velocity=None)
        self.status_label.setText("done!")

    def parse_midi(self):
        """
        Parse midi file
        """
        from score_alignment.lilypond_note_coords.MidiParser import MidiParser
        
        self.status_label.setText("Parsing midi ...")
        
        pattern = self.folder_name + "/audio/*.midi"
        for midi_file_path in glob.glob(pattern):
            print "Processing", midi_file_path

            # get file names and directories
            directory = os.path.dirname(midi_file_path)
            file_name = os.path.basename(midi_file_path).split('.midi')[0]
            audio_file_path = os.path.join(directory, file_name + '.flac')
            spec_file_path = os.path.join(directory.replace("/audio", "/spec"), file_name + '_spec.npy')
            onset_file_path = os.path.join(directory.replace("/audio", "/spec"), file_name + '_onsets.npy')
            
            # check if to compute spectrogram
            if not self.checkBox_computeSpec.isChecked():
                audio_file_path = None
                self.spec = np.load(spec_file_path)
            
            # parse midi file
            midi_parser = MidiParser(show=self.checkBox_showSpec.isChecked())
            Spec, self.onsets = midi_parser.process(midi_file_path, audio_file_path)

            # save data
            if self.checkBox_computeSpec.isChecked():
                self.spec = Spec
                np.save(spec_file_path, self.spec)
            np.save(onset_file_path, self.onsets)
        
        # set number of onsets in gui
        self.lineEdit_nOnsets.setText(str(len(self.onsets)))
        
        self.status_label.setText("done!")

    def load_sheet(self):
        """
        Load sheet image
        """
        self.status_label.setText("Loading sheet ...")

        self.sheet_pages = []
        self.page_coords = []
        self.page_rois = []
        self.page_systems = []

        # prepare paths
        sheet_dir = os.path.join(self.folder_name, "sheet")
        img_files = np.sort(glob.glob(sheet_dir + "/*.*"))

        # load data
        n_pages = len(img_files)
        for i in xrange(n_pages):
            self.sheet_pages.append(cv2.imread(img_files[i], 0))
            
            self.page_coords.append(np.zeros((0, 2)))
            self.page_rois.append([])
            self.page_systems.append(np.zeros((0, 4, 2)))

        self.spinBox_page.setMaximum(n_pages - 1)
        
        self.update_sheet_statistics()
        
        self.status_label.setText("done!")

    def load_coords(self):
        """ Load coordinates """
        self.load_system_coords()
        self.load_note_coords()

    def load_system_coords(self):
        """ Load system coordinates """
        self.status_label.setText("Loading system coords ...")

        # prepare paths
        coord_dir = os.path.join(self.folder_name, "coords")
        coord_files = np.sort(glob.glob(coord_dir + "/systems_*.npy"))

        # load data
        if len(coord_files) > 0:
            for i in xrange(self.n_pages):
                if os.path.exists(coord_files[i]):
                    self.page_systems[i] = np.load(coord_files[i])

        # convert systems to rois
        self.systems_to_rois()

        self.status_label.setText("done!")

    def load_note_coords(self):
        """ Load note coordinates """
        self.status_label.setText("Loading coords ...")

        # prepare paths
        coord_dir = os.path.join(self.folder_name, "coords")
        coord_files = np.sort(glob.glob(coord_dir + "/notes_*.npy"))

        # load data
        if len(coord_files) > 0:
            for i in xrange(self.n_pages):
                if os.path.exists(coord_files[i]):
                    page_coords = np.load(coord_files[i])
                    self.page_coords[i] = page_coords

        self.update_sheet_statistics()
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
        """ Sort note coordinates by rows """
        
        for page_id in xrange(self.n_pages):
            page_coords = self.page_coords[page_id]
            page_rois = self.page_rois[page_id]

            page_coords = sort_by_roi(page_coords, page_rois)

            self.page_coords[page_id] = page_coords
        
    def update_sheet_statistics(self):
        """ Compute sheet statistics """

        self.n_pages = len(self.sheet_pages)
        self.n_systems = np.sum([len(s) for s in self.page_systems])
        self.n_coords = np.sum([len(c) for c in self.page_coords])

        self.lineEdit_nPages.setText(str(self.n_pages))
        self.lineEdit_nSystems.setText(str(self.n_systems))
        self.lineEdit_nCoords.setText(str(self.n_coords))
    
    def save_coords(self):
        """ Save changed sheet coords """
        coord_dir = os.path.join(self.folder_name, 'coords')
        for i in xrange(len(self.page_coords)):
            coord_file = os.path.join(coord_dir, "notes_%02d.npy" % (i + 1))
            np.save(coord_file, self.page_coords[i])

            coord_file = os.path.join(coord_dir, "systems_%02d.npy" % (i + 1))
            np.save(coord_file, self.page_systems[i])

    def edit_coords(self):
        """ Edit sheet elements """

        # show sheet image along coordinates
        self.fig = plt.figure("Sheet Editor")

        # init events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.plot_sheet()

    def plot_sheet(self):
        """
        Plot sheet image along with coordinates
        """

        # get data of current page
        page_id = self.spinBox_page.value()

        self.fig.clf()

        # plot sheet image
        ax = plt.subplot(111)
        plt.subplots_adjust(top=0.98, bottom=0.05)
        plt.imshow(self.sheet_pages[page_id], cmap=plt.cm.gray, interpolation='nearest')
        plt.xlim([0, self.sheet_pages[page_id].shape[1] - 1])
        plt.ylim([self.sheet_pages[page_id].shape[0] - 1, 0])
        plt.xlabel(self.sheet_pages[page_id].shape[1])
        plt.ylabel(self.sheet_pages[page_id].shape[0])

        # plot note coordinates
        if self.checkBox_showCoords.isChecked():
            plt.plot(self.page_coords[page_id][:, 1], self.page_coords[page_id][:, 0], 'co')

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

        plt.draw()
        plt.pause(0.1)

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
        if event.button == 3:
            
            plt.figure("Spectrogram")
            plt.clf()
            plt.imshow(self.spec, aspect='auto', origin='lower')
            
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

            # find closets system
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
                    print "Removed system with id:", i
                    break

            self.systems_to_rois()
        
        # remove note position
        if self.radioButton_deletNote.isChecked():
            
            # find closets note
            dists = pairwise_distances(clicked, self.page_coords[page_id])
            selection = np.argmin(dists)
            
            # remove coordinate
            self.page_coords[page_id] = np.delete(self.page_coords[page_id], selection, axis=0)
            print "Removed note with id:", selection
        
        # add note position
        if self.radioButton_addNote.isChecked():
            
            # find closets note
            if self.page_coords[page_id].shape[0] > 0:
                dists = pairwise_distances(clicked, self.page_coords[page_id])
                selection = np.argmin(dists)
            else:
                selection = -1
            
            self.page_coords[page_id] = np.insert(self.page_coords[page_id], selection + 1, clicked, axis=0)
        
        # update sheet statistics
        self.update_sheet_statistics()        
        
        # refresh view
        self.sort_note_coords()
        self.plot_sheet()

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
        from omr.models import system_detector as system_model
        from lasagne_wrapper.network import SegmentationNetwork
        from omr.omr_app import OpticalMusicRecognizer

        # initialize note detection neural network
        dump_file = "/home/matthias/experiments/omr/note_detector/params.pkl"
        net = note_model.build_model()
        note_net = SegmentationNetwork(net, print_architecture=False)
        note_net.load(dump_file)

        # initialize system detection neural network
        dump_file = "/home/matthias/experiments/omr/system_detector/params.pkl"
        net = system_model.build_model()
        system_net = SegmentationNetwork(net, print_architecture=False)
        system_net.load(dump_file)

        # initialize omr system
        self.omr = OpticalMusicRecognizer(note_detector=note_net, system_detector=system_net)

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

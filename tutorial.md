
Sheet Manager
=============

The Sheet Manager tool produces the multimodal sheet music dataset (MSMD)
from scratch. In this case, "scratch" means the Mutopia archive.
Start by getting Mutopia from GitHub:

```
    git clone https://github.com/MutopiaProject/MutopiaProject.git
```

To replicate our experiments, you will need the same state of the repository:

```
    cd MutopiaProject
    git checkout e325d76f7eb728aebff056822b286be8f6b06aac
```

Initialize the Mutopia data root var, which we'll be using in the rest of the
tutorial:

```
    cd MutopiaProject/ftp/
    MUTO_ROOT=`pwd`
```

And we also initialize our MSMD directory:

```
    mkdir -p $MSMD
```

In the big picture, there are two steps:

* Export LilyPond files of piano music from Mutopia to MSMD
* Process the exported files to create the multimodal data for the piece

Mutopia export
--------------

Step 1 creates a directory for each piece inside `$MSMD` with a single
LilyPond file. This file then acts as the authority encoding for the given
piece in MSMD. The ``mutopia/process_mutopia.py`` script wraps all the steps
necessary for extracting all piano pieces in Mutopia, exporting this `*.ly`
file. Run the script as:

```
    python process_mutopia.py --mutopia_root $MUTO_ROOT --output_root $MSMD
```

The script first creates a simplified data model of the Mutopia collection.
(This is the time-consuming part of the script's runtime.) Then, it
resolves which LilyPond files correspond to actual pieces (combines header
and `\include` information), applies some LilyPond preprocessing (process
`\include` statements, ensure point-and-click functionality is on, removes
MIDI repeat unfolding statements, etc.), and exports the resulting files
into `$MSMD` subdirs. (The names of the subdirectories are built from the
Mutopia header fields: `mutopiacomposer`, `mutopiaopus`, and the name
of the piece file.)

For more options, run `python process_mutopia.py -h`.


MSMD generation
---------------

After the `*.ly` authority encodings for each piece are extracted from Mutopia,
we use them to generate its score(s) and performance(s), the score views
(images, coords, MuNG), the performance features (MIDI matrix, onsets list,
note event matrix, and spectrogram), and the alignment between MuNG objects
and MIDI note events.

Each piece in MSMD is processed separately using the `sheet_manager_app.py`
and the `SheetManager` class therein. There is an interactive mode
for visualizing results and debugging the extraction pipeline, and a batch
mode for processing multiple pieces in one run.

To process the entire dataset, run:

```
    python sheet_manager_app.py -d $MSMD --all
```

which is a shortcut for:

```
    python sheet_manager_app.py -d $MSMD -p `ls $MSMD`
```

implying that you can process specific pieces by supplying a list of their
names (the directory names in `$MSMD`) to the `-p` argument of Sheet Manager.

The batch mode ignores errors (the piece is simply left in the state in which
the score/performance generating pipeline encountered the error), recording
them in a log. There are some 650+ piano pieces in Mutopia; running the whole
pipeline with `--all` takes several hours, depending on your CPU, and it can
of course be parallelized.



Extracting alignment
====================

### give piece dir
collection_dir = '/home/matthias/cp/data/msmd'
piece_name = 'BachCPE__cpe-bach-rondo__cpe-bach-rondo'
piece_dir = os.path.join(collection_dir, piece_name)

app = QtGui.QApplication(sys.argv)
mgr = SheetManager()

# Piece loading

from sheet_manager.data_model.piece import Piece
piece = Piece(root=collection_dir, name=piece_name)
score = piece.load_score(piece.available_scores[0])
performance = piece.load_performance(piece.available_performances[0], require_audio=False)

# Running the alignment procedure

from sheet_manager.alignments import align_score_to_performance
alignment = align_score_to_performance(score, performance)



# Inspect the alignemnt

m_objid, e_idx = alignment[0]

note_events = performance.load_note_events()
mungos = score.load_mungos()
mdict = {m.objid: m for m in mungos}


# To retrieve also the page where the individual MuNGs are
mungos_per_page = score.load_mungos(by_page=True)
mungo_to_page = {}
for i, ms in enumerate(mungos_per_page):
	for m in ms:
  	mungo_to_page[m.objid] = i



m, e = mdict[m_objid], note_events[e_idx]
# ...this is now the aligned pair of the MuNG object representing the note(head) in the score,
#    and the MIDI note event.



# Recording the alignment inside the MuNG objects: SheetManager.update_mung_alignment()
# This stores the alignment persistently.
onset_frame = notes_to_onsets([e], dt=1.0 / FPS)
m.data['{0}_onset_seconds'.format(performance.name)] = e[0]
m.data['{0}_onset_frame'.format(performance.name)] = int(onset_frame)






# Finding the position of a MuNG object in the images:
m_page = mungo_to_page[m.objid]
t, l, b, r = m.bounding_box   # Integers!
cx, cy = m.middle


# BTW: Minimal integer bounding box containing the floating-point bbox:
it, il, ib, ir = CropObject.bbox_to_integer_bounds(ft, fl, fb, fr)


images = score.load_images()
img = images[m_page]



### Compute system coordinates

from sheet_music.alignments import detect_system_regions_ly
systems_as_corners = detect_system_regions_ly(img)

# To update the MuNG staff objects, see SheetManager.detect_systems() method.
# Then, persist the MuNGs back to the score: see SheetManager.save_mung()




# Getting the noteheads in one system:
staff_noteheads = [mdict[i] for i in staff_mung.inlinks]





### unroll all systems from all three pages to one very long system (adapt note coordinates)


# load performance
# load alignment
# take onset/frame
# onset points to ID in the notehead list


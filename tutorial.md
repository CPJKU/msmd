
Sheet Manager
=============

The Sheet Manager tool produces the multimodal sheet music dataset (MSMD)
from scratch. In this case, "scratch" means the Mutopia archive.
Start by getting Mutopia from GitHub:

```
    git clone https://github.com/MutopiaProject/MutopiaProject.git
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

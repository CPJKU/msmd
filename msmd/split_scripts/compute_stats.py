
import yaml
import numpy as np
from audio_sheet_retrieval.utils.data_pools import prepare_piece_data


compute_note_counts = False


if __name__ == "__main__":
    """ main """

    collection_dir = '/media/matthias/Data/msmd/'
    piece_file = "/home/matthias/cp/src/msmd/msmd/splits/all_pieces.yaml"
    count_file = piece_file.replace(".yaml", "_counts.yaml")

    # compute note counts for piece list
    if compute_note_counts:

        with open(piece_file, 'rb') as hdl:
            pieces = yaml.load(hdl)

        note_counts = dict()
        for i, piece_name in enumerate(pieces["success"]):
            print "%d / %d" % (i+1, len(pieces["success"]))
            _, _, piece_o2c_maps = prepare_piece_data(collection_dir, piece_name)
            count = len(piece_o2c_maps[0])
            note_counts[piece_name] = count

        with open(count_file, 'wb') as hdl:
            yaml.dump(note_counts, hdl)

    # load note counts
    else:
        with open(count_file, 'rb') as hdl:
            note_counts = yaml.load(hdl)


    # compute stats
    # -------------

    n_pieces = len(note_counts)
    n_onsets = np.sum(note_counts.values())

    print "%d pieces with %d onsets available" % (n_pieces, n_onsets)

    piece_names = np.asarray(note_counts.keys())
    counts = np.asarray(note_counts.values())

    sorted_idx = np.argsort(counts)
    piece_names = piece_names[sorted_idx]
    counts = counts[sorted_idx]

    # get composers
    composer_names = [p.split("__")[0] for p in piece_names]
    composers, comp_counts = np.unique(composer_names, return_counts=True)

    sorted_idx = np.argsort(comp_counts)[::-1]
    composers = composers[sorted_idx]
    comp_counts = comp_counts[sorted_idx]

    n_more_than_one = 0
    for i, comp in enumerate(composers):

        if comp_counts[i] > 1:
            n_more_than_one += 1

        print i+1, comp, comp_counts[i]

    print "Number of composers:", len(composers)
    print "Number of composers with more than one piece:", n_more_than_one


import yaml


if __name__ == "__main__":
    """ main """

    collection_dir = '/media/matthias/Data/msmd/'
    piece_file = "/home/matthias/cp/src/sheet_manager/sheet_manager/splits/all_pieces_aug.yaml"

    with open(piece_file, 'rb') as hdl:
        pieces = yaml.load(hdl)

    for p1 in pieces["problems"]:
        if p1 in pieces["success"]:
            pieces["success"].remove(p1)

    with open(piece_file, 'wb') as hdl:
        pieces = yaml.dump(pieces, hdl, default_flow_style=False)

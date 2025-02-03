import os
import math
import json
import numpy as np
from argparse import ArgumentParser



def coords_to_idxs(coords):
    xmin, ymin = np.min(coords, 0)
    xmax, ymax = np.max(coords, 0)

    xmin_idx, ymin_idx = math.floor(xmin / 960), math.floor(ymin / 960)
    xmax_idx, ymax_idx = math.ceil(xmax / 960), math.ceil(ymax / 960)

    return xmin_idx, ymin_idx, xmax_idx, ymax_idx

def process_case(case_path):
    pyramid_path = os.path.join(case_path, "pyramid", "tiles_files")
    print("pyramid_path", pyramid_path)
    if not os.path.exists(pyramid_path):
        return

    levels = [int(level_dir) for level_dir in os.listdir(pyramid_path) if os.path.isdir(os.path.join(pyramid_path, level_dir))]
    print("levels:", levels)
    levels.sort()
    print("sorted levels:", levels)
    if not levels:
        return

    highest_level = levels[-1]
    highest_level_path = os.path.join(pyramid_path, str(highest_level))
    print("highest_level_path:", highest_level_path)
    idxs = []
    for jpeg_file in os.listdir(highest_level_path):
        if jpeg_file.endswith('.webp'):
            # idx = tuple(map(int, jpeg_file.split('.')[0].split('_')[1:3]))
            try:
                # Split and convert to integer tuple
                parts = jpeg_file.split('.')[0].split('_')
                if len(parts) >= 2:
                    idx = tuple(map(int, parts[:2]))
                    print("jpeg_file idx:", idx)
                    idxs.append(idx)
                else:
                      print(f"Unexpected filename format: {jpeg_file}")
            except ValueError as e:
                print(f"Error processing filename {jpeg_file}: {e}")

    idxs = np.array(idxs)
    nitems = len(idxs)
    print("number of items:", nitems)
    annot_idxs = np.zeros((nitems, 1), dtype=np.int32)

    annot_file = os.path.join(case_path, 'annot.json')
    if os.path.exists(annot_file):
        with open(annot_file, 'r') as f:
            annot_data = json.load(f)

        try:
            seg_items = annot_data[0][1]["children"]

            annot_idx_list = []
            for seg_item in seg_items:
                coords = np.zeros((4, 2))
                for cntr, segment in enumerate(seg_item[1]["segments"]):
                    if cntr >= 4:
                        print(f"Skipping out-of-bounds segment index {cntr} in {case_path}")
                        continue
                    coords[cntr, :] = segment[0]

                xmin_idx, ymin_idx, xmax_idx, ymax_idx = coords_to_idxs(coords)
                for c in range(xmin_idx, xmax_idx + 1):
                    for r in range(ymin_idx, ymax_idx + 1):
                        annot_idx_list.append((c, r))

            for i in range(nitems):
                if tuple(idxs[i]) in annot_idx_list:
                    annot_idxs[i] = 1

        except Exception as e:
            print(f"Error processing {case_path}: {e}")
            return

    annot_idxs.tofile(os.path.join(pyramid_path, f'level-{highest_level}_annot_idxs.32i'))

def process_directory(directory):
    print("directory:", directory)
    for root, dirs, files in os.walk(directory):
        if "annot.json" in files:
            print("root:", root)
            process_case(root)

if __name__ == "__main__":
    parser = ArgumentParser(description="Label webp images based on annotations")
    parser.add_argument(dest="vx_client_dir", help="Directory with all vx_client_data", metavar="$VX_DIR$")
    args = parser.parse_args()

    vx_client_dir = args.vx_client_dir
    process_directory(vx_client_dir)

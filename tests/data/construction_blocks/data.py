"""
Generates test data.
"""

import tifffile as tif
import numpy as np
import pandas as pd
from pathlib import Path


TEST_DATA = r"D:\Code\repos\vodex\data\test"


def create_test_movie():
    """
    Makes a movie with labeled frames, slices in the volumes and conditions.
    """
    # load images
    c1_img = tif.imread(Path(TEST_DATA, "construction_blocks", "zero_frame_c1.tif")).astype(bool)
    c2_img = tif.imread(Path(TEST_DATA, "construction_blocks", "zero_frame_c2.tif")).astype(bool)
    c3_img = tif.imread(Path(TEST_DATA, "construction_blocks", "zero_frame_c3.tif")).astype(bool)
    s_img = tif.imread(Path(TEST_DATA, "construction_blocks", "zero_frame_square.tif")).astype(bool)
    c_img = tif.imread(Path(TEST_DATA, "construction_blocks", "zero_frame_circle.tif")).astype(bool)
    f_img = tif.imread(Path(TEST_DATA, "construction_blocks", "numbered_frames.tif")).astype(bool)
    # load annotation
    df = pd.read_csv(Path(TEST_DATA, "construction_blocks", "recepie.csv"))
    # map annotation to images
    img_dict = {"c1": c1_img,
                "c2": c2_img,
                "c3": c3_img,
                "s": s_img,
                "c": c_img}
    # create movie
    movie = np.zeros((len(df), c1_img.shape[1], c1_img.shape[2]))
    for index, row in df.iterrows():
        # pick the stimuli and slice
        movie[index, :] = img_dict[row['c_stim']][row['slice']] + \
                          img_dict[row['s_stim']][0] + \
                          f_img[index]
        # invert if light is on
        if row['light'] == "on":
            movie[index, :] = 1 - movie[index, :]

    # save full movie image
    tif.imwrite(Path(TEST_DATA, "test_movie.tif"), np.expand_dims(movie.astype(np.uint16), axis=1), imagej=True)

    # split movie in 3 chunks and write to folder
    # (frame IDs per file : [1,7],[8,25],[26,42] )
    movie0, movie1, movie2 = movie[0:7], movie[7:25], movie[25:42]
    tif.imwrite(Path(TEST_DATA, "test_movie", "mov0.tif"), np.expand_dims(movie0.astype(np.uint16), axis=1),
                imagej=True)
    tif.imwrite(Path(TEST_DATA, "test_movie", "mov1.tif"), np.expand_dims(movie1.astype(np.uint16), axis=1),
                imagej=True)
    tif.imwrite(Path(TEST_DATA, "test_movie", "mov2.tif"), np.expand_dims(movie2.astype(np.uint16), axis=1),
                imagej=True)

    # write frames 1. 2. 41. 42 only :
    frames = movie[[0, 1, 40, 41]]
    tif.imwrite(Path(TEST_DATA, "frames_1_2_41_42.tif"), np.expand_dims(frames.astype(np.uint16), axis=1),
                imagej=True)

    # write volume
    volume_frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    volumes = movie[volume_frames]
    tif.imwrite(Path(TEST_DATA, "volumes_0_1.tif"),
                np.expand_dims(volumes.astype(np.uint16).reshape((2, 10, 200, 200)), axis=1),
                imagej=True)

    # write half-volume
    volume_frames = [0, 1, 2, 3, 4,
                     10, 11, 12, 13, 14]
    volumes = movie[volume_frames]
    tif.imwrite(Path(TEST_DATA, "half_volumes_0_1.tif"),
                np.expand_dims(volumes.astype(np.uint16).reshape((2, 5, 200, 200)), axis=1),
                imagej=True)

    # write tail-volume
    volume_frames = [40,41]
    volumes = movie[volume_frames]
    tif.imwrite(Path(TEST_DATA, "volumes_tail.tif"),
                np.expand_dims(volumes.astype(np.uint16).reshape((1, 2, 200, 200)), axis=1),
                imagej=True)


if __name__ == "__main__":
    create_test_movie()

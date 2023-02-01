"""
Tests for the `vodex.loaders` module.
"""
import numpy as np
from pathlib import Path
import pytest
import tifffile as tif

from vodex import *


TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')

FULL_MOVIE = Path(TEST_DATA, "test_movie.tif")
SPLIT_MOVIES = [Path(TEST_DATA, "test_movie", mov) for mov in
                ["mov0.tif", "mov1.tif", "mov2.tif"]]
FRAMES_1_2_41_42 = tif.imread(Path(TEST_DATA, 'loader_test', "frames_1_2_41_42.tif").as_posix())
VOLUMES_0_1 = tif.imread(Path(TEST_DATA, 'loader_test', "volumes_1_2.tif").as_posix())
HALF_VOLUMES_0_1 = tif.imread(Path(TEST_DATA, 'loader_test', "half_volumes_1_2.tif").as_posix())


@pytest.fixture
def tiff_loader():
    return TiffLoader(FULL_MOVIE)

def test_eq():
    loader1 = TiffLoader(FULL_MOVIE)
    loader2 = TiffLoader(FULL_MOVIE)
    assert loader1 == loader2
    assert loader2 == loader1
    assert loader1.__eq__("TiffLoader") == NotImplemented


def test_get_frames_in_file(tiff_loader):
    assert tiff_loader.get_frames_in_file(FULL_MOVIE) == 42

def test_get_frame_size(tiff_loader):
    f_size = tiff_loader.get_frame_size(FULL_MOVIE)
    assert f_size == (200, 200)

def test_get_frame_dtype(tiff_loader):
    data_type = tiff_loader.get_frame_dtype(FULL_MOVIE)
    assert data_type == np.uint16

def test_load_frames_one_file(tiff_loader):
    frames = [0, 1, 40, 41]
    print("Must show a progress bar:")
    f_img = tiff_loader.load_frames(frames, [FULL_MOVIE] * 4)
    assert f_img.shape == (4, 200, 200)
    print("Must show 'Loading from file' and one file:")
    f_img = tiff_loader.load_frames(frames, [FULL_MOVIE] * 4, show_file_names=True)
    assert f_img.shape == (4, 200, 200)
    assert (f_img == FRAMES_1_2_41_42).all()

def test_load_frames_many_files(tiff_loader):
    frames = [0, 1, 15, 16]
    print("Must show a progress bar:")
    files = [SPLIT_MOVIES[0], SPLIT_MOVIES[0],
             SPLIT_MOVIES[2], SPLIT_MOVIES[2]]
    f_img = tiff_loader.load_frames(frames, files)
    assert f_img.shape == (4, 200, 200)
    print("Must show 'Loading from file' and two files:")
    f_img = tiff_loader.load_frames(frames, files, show_file_names=True)
    assert f_img.shape == (4, 200, 200)
    assert (f_img == FRAMES_1_2_41_42).all()

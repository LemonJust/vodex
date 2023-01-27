"""
Tests for the `vodex.loaders` module.
"""
import numpy as np
from pathlib import Path
import pytest
import tifffile as tif

from vodex import *


TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')

full_movie = Path(TEST_DATA, "test_movie.tif")
split_movies = [Path(TEST_DATA, "test_movie", mov) for mov in
                ["mov0.tif", "mov1.tif", "mov2.tif"]]
frames_1_2_41_42 = tif.imread(Path(TEST_DATA, 'loader_test', "frames_1_2_41_42.tif"))
volumes_0_1 = tif.imread(Path(TEST_DATA, 'loader_test', "volumes_1_2.tif"))
half_volumes_0_1 = tif.imread(Path(TEST_DATA, 'loader_test', "half_volumes_1_2.tif"))


# TODO : use fixtures

class TestTiffLoader:

    def test_eq(self):
        loader1 = TiffLoader(full_movie)
        loader2 = TiffLoader(full_movie)
        assert loader1 == loader2
        assert loader2 == loader1

    def test_get_frames_in_file(self):
        loader = TiffLoader(full_movie)
        n_frames = loader.get_frames_in_file(full_movie)
        assert n_frames == 42

    def test_get_frame_size(self):
        loader = TiffLoader(full_movie)
        f_size = loader.get_frame_size(full_movie)
        assert f_size == (200, 200)

    def test_get_frame_dtype(self):
        loader = TiffLoader(full_movie)
        data_type = loader.get_frame_dtype(full_movie)
        assert data_type == np.uint16

    def test_load_frames_one_file(self):
        loader = TiffLoader(full_movie)
        frames = [0, 1, 40, 41]
        print("Must show a progress bar:")
        f_img = loader.load_frames(frames, [full_movie] * 4)
        assert f_img.shape == (4, 200, 200)
        print("Must show 'Loading from file' and one file:")
        f_img = loader.load_frames(frames, [full_movie] * 4, show_file_names=True)
        assert f_img.shape == (4, 200, 200)
        assert (f_img == frames_1_2_41_42).all()

    def test_load_frames_many_files(self):
        loader = TiffLoader(full_movie)
        frames = [0, 1, 15, 16]
        print("Must show a progress bar:")
        files = [split_movies[0], split_movies[0],
                 split_movies[2], split_movies[2]]
        f_img = loader.load_frames(frames, files)
        assert f_img.shape == (4, 200, 200)
        print("Must show 'Loading from file' and two files:")
        f_img = loader.load_frames(frames, files, show_file_names=True)
        assert f_img.shape == (4, 200, 200)
        assert (f_img == frames_1_2_41_42).all()


class TestImageLoader:

    def test_eq(self):
        loader1 = ImageLoader(full_movie)
        loader2 = ImageLoader(full_movie)
        assert loader1 == loader2
        assert loader2 == loader1

    def test_init_loader(self):
        tif_loader = TiffLoader(full_movie)
        loader = ImageLoader(full_movie).loader
        assert loader == tif_loader

    # def test_get_frames_in_file(self):
    #     loader = ImageLoader(full_movie)
    #     n_frames = loader.get_frames_in_file(full_movie)
    #     assert n_frames == 42

    def test_get_frame_size(self):
        loader = ImageLoader(full_movie)
        f_size = loader.get_frame_size(full_movie)
        assert f_size == (200, 200)

    def test_load_frames_one_file(self):
        loader = ImageLoader(full_movie)

        frames = [0, 1, 40, 41]
        files = [full_movie] * 4

        print("Must show a progress bar:")
        f_img = loader.load_frames(frames, files)
        assert f_img.shape == (4, 200, 200)

        print("Must show 'Loading from file' and one file:")
        f_img = loader.load_frames(frames, files, show_file_names=True)
        assert f_img.shape, (4, 200, 200)
        assert (f_img == frames_1_2_41_42).all()

    def test_load_frames_many_files(self):
        loader = ImageLoader(full_movie)

        frames = [0, 1, 15, 16]
        files = [split_movies[0], split_movies[0],
                 split_movies[2], split_movies[2]]

        print("Must show a progress bar:")
        f_img = loader.load_frames(frames, files)
        assert f_img.shape == (4, 200, 200)

        print("Must show 'Loading from file' and two files:")
        f_img = loader.load_frames(frames, files, show_file_names=True)
        assert f_img.shape == (4, 200, 200)
        assert (f_img == frames_1_2_41_42).all()

    def test_load_volumes_full(self):
        loader = ImageLoader(full_movie)
        # TODO : check all the places for consistency n volumes 1 2 meaning 0 1 actually :(

        frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        volumes = [0] * 10
        volumes.extend([1] * 10)
        files = [full_movie] * 20

        v_img = loader.load_volumes(frames, files, volumes)
        assert v_img.shape == (2, 10, 200, 200)
        assert (v_img == volumes_0_1).all()

    def test_load_volumes_half(self):
        loader = ImageLoader(full_movie)

        frames = [0, 1, 2, 3, 4,
                  10, 11, 12, 13, 14]
        volumes = [1] * 5
        volumes.extend([2] * 5)
        files = [full_movie] * 10

        v_img = loader.load_volumes(frames, files, volumes)
        assert v_img.shape == (2, 5, 200, 200)
        assert (v_img == half_volumes_0_1).all()

        # now let's make sure it breaks when we ask for different number of slices per volume
        volumes = [1] * 6
        volumes.extend([2] * 4)
        files = [full_movie] * 10
        with pytest.raises(AssertionError):
            loader.load_volumes(frames, files, volumes)

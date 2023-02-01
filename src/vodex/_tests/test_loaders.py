"""
Tests for the `vodex.loaders` module.
"""
import pytest
from vodex import *
from .conftest import (FULL_MOVIE,
                       SPLIT_MOVIE,
                       FRAMES_1_2_41_42,
                       VOLUMES_0_1,
                       HALF_VOLUMES_0_1)


class TestImageLoader:

    def test_eq(self):
        loader1 = ImageLoader(FULL_MOVIE)
        loader2 = ImageLoader(FULL_MOVIE)
        assert loader1 == loader2
        assert loader2 == loader1

    def test_init_loader(self):
        tif_loader = TiffLoader(FULL_MOVIE)
        loader = ImageLoader(FULL_MOVIE).loader
        assert loader == tif_loader

    def test_get_frame_size(self):
        loader = ImageLoader(FULL_MOVIE)
        f_size = loader.get_frame_size(FULL_MOVIE)
        assert f_size == (200, 200)

    def test_load_frames_one_file(self):
        loader = ImageLoader(FULL_MOVIE)

        frames = [0, 1, 40, 41]
        files = [FULL_MOVIE] * 4

        print("Must show a progress bar:")
        f_img = loader.load_frames(frames, files)
        assert f_img.shape == (4, 200, 200)

        print("Must show 'Loading from file' and one file:")
        f_img = loader.load_frames(frames, files, show_file_names=True)
        assert f_img.shape, (4, 200, 200)
        assert (f_img == FRAMES_1_2_41_42).all()

    def test_load_frames_many_files(self):
        loader = ImageLoader(FULL_MOVIE)

        frames = [0, 1, 15, 16]
        files = [SPLIT_MOVIE[0], SPLIT_MOVIE[0],
                 SPLIT_MOVIE[2], SPLIT_MOVIE[2]]

        print("Must show a progress bar:")
        f_img = loader.load_frames(frames, files)
        assert f_img.shape == (4, 200, 200)

        print("Must show 'Loading from file' and two files:")
        f_img = loader.load_frames(frames, files, show_file_names=True)
        assert f_img.shape == (4, 200, 200)
        assert (f_img == FRAMES_1_2_41_42).all()

    def test_load_volumes_full(self):
        loader = ImageLoader(FULL_MOVIE)
        # TODO : check all the places for consistency n volumes 1 2 meaning 0 1 actually :(

        frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        volumes = [0] * 10
        volumes.extend([1] * 10)
        files = [FULL_MOVIE] * 20

        v_img = loader.load_volumes(frames, files, volumes)
        assert v_img.shape == (2, 10, 200, 200)
        assert (v_img == VOLUMES_0_1).all()

    def test_load_volumes_half(self):
        loader = ImageLoader(FULL_MOVIE)

        frames = [0, 1, 2, 3, 4,
                  10, 11, 12, 13, 14]
        volumes = [1] * 5
        volumes.extend([2] * 5)
        files = [FULL_MOVIE] * 10

        v_img = loader.load_volumes(frames, files, volumes)
        assert v_img.shape == (2, 5, 200, 200)
        assert (v_img == HALF_VOLUMES_0_1).all()

        # now let's make sure it breaks when we ask for different number of slices per volume
        volumes = [1] * 6
        volumes.extend([2] * 4)
        files = [FULL_MOVIE] * 10
        with pytest.raises(AssertionError):
            loader.load_volumes(frames, files, volumes)

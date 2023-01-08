"""
Tests for the `vodex.loaders` module.
"""
import unittest
import tifffile as tif

from vodex import *

# package_dir = Path(__file__).parents[1]
# TEST_DATA = Path(package_dir, 'data', 'test')


class TestTiffLoader(unittest.TestCase):
    full_movie = Path(TEST_DATA, "test_movie.tif")
    split_movies = [Path(TEST_DATA, "test_movie", mov) for mov in
                    ["mov0.tif", "mov1.tif", "mov2.tif"]]
    frames_1_2_41_42 = tif.imread(Path(TEST_DATA, "frames_1_2_41_42.tif"))

    def test_eq(self):
        loader1 = TiffLoader(self.full_movie)
        loader2 = TiffLoader(self.full_movie)
        self.assertEqual(loader1, loader2)
        self.assertEqual(loader2, loader1)

    def test_get_frames_in_file(self):
        loader = TiffLoader(self.full_movie)
        n_frames = loader.get_frames_in_file(self.full_movie)
        self.assertEqual(n_frames, 42)

    def test_get_frame_size(self):
        loader = TiffLoader(self.full_movie)
        f_size = loader.get_frame_size(self.full_movie)
        self.assertEqual(f_size, (200, 200))

    def test_get_frame_dtype(self):
        loader = TiffLoader(self.full_movie)
        data_type = loader.get_frame_dtype(self.full_movie)
        self.assertEqual(data_type, np.uint16)

    def test_load_frames_one_file(self):
        loader = TiffLoader(self.full_movie)
        frames = [0, 1, 40, 41]
        print("Must show a progress bar:")
        f_img = loader.load_frames(frames, [self.full_movie] * 4)
        self.assertEqual(f_img.shape, (4, 200, 200))
        print("Must show 'Loading from file' and one file:")
        f_img = loader.load_frames(frames, [self.full_movie] * 4, show_file_names=True)
        self.assertEqual(f_img.shape, (4, 200, 200))
        self.assertTrue(np.all(np.equal(f_img, self.frames_1_2_41_42)))

    def test_load_frames_many_files(self):
        loader = TiffLoader(self.full_movie)
        frames = [0, 1, 15, 16]
        print("Must show a progress bar:")
        files = [self.split_movies[0], self.split_movies[0],
                 self.split_movies[2], self.split_movies[2]]
        f_img = loader.load_frames(frames, files)
        self.assertEqual(f_img.shape, (4, 200, 200))
        print("Must show 'Loading from file' and two files:")
        f_img = loader.load_frames(frames, files, show_file_names=True)
        self.assertEqual(f_img.shape, (4, 200, 200))
        self.assertTrue(np.all(np.equal(f_img, self.frames_1_2_41_42)))


class TestImageLoader(unittest.TestCase):
    data_dir_full = TEST_DATA

    full_movie = Path(TEST_DATA, "test_movie.tif")
    split_movies = [Path(TEST_DATA, "test_movie", mov) for mov in
                    ["mov0.tif", "mov1.tif", "mov2.tif"]]

    frames_1_2_41_42 = tif.imread(Path(TEST_DATA, "frames_1_2_41_42.tif"))
    volumes_0_1 = tif.imread(Path(TEST_DATA, "volumes_1_2.tif"))
    half_volumes_0_1 = tif.imread(Path(TEST_DATA, "half_volumes_1_2.tif"))

    def test_eq(self):
        loader1 = ImageLoader(self.full_movie)
        loader2 = ImageLoader(self.full_movie)
        self.assertEqual(loader1, loader2)
        self.assertEqual(loader2, loader1)

    def test_init_loader(self):
        tif_loader = TiffLoader(self.full_movie)
        loader = ImageLoader(self.full_movie).loader
        self.assertEqual(loader, tif_loader)

    def test_get_frames_in_file(self):
        loader = ImageLoader(self.full_movie)
        n_frames = loader.get_frames_in_file(self.full_movie)
        self.assertEqual(n_frames, 42)

    def test_get_frame_size(self):
        loader = ImageLoader(self.full_movie)
        f_size = loader.get_frame_size(self.full_movie)
        self.assertEqual(f_size, (200, 200))

    def test_load_frames_one_file(self):
        loader = ImageLoader(self.full_movie)

        frames = [0, 1, 40, 41]
        files = [self.full_movie] * 4

        print("Must show a progress bar:")
        f_img = loader.load_frames(frames, files)
        self.assertEqual(f_img.shape, (4, 200, 200))

        print("Must show 'Loading from file' and one file:")
        f_img = loader.load_frames(frames, files, show_file_names=True)
        self.assertEqual(f_img.shape, (4, 200, 200))
        self.assertTrue(np.all(np.equal(f_img, self.frames_1_2_41_42)))

    def test_load_frames_many_files(self):
        loader = ImageLoader(self.full_movie)

        frames = [0, 1, 15, 16]
        files = [self.split_movies[0], self.split_movies[0],
                 self.split_movies[2], self.split_movies[2]]

        print("Must show a progress bar:")
        f_img = loader.load_frames(frames, files)
        self.assertEqual(f_img.shape, (4, 200, 200))

        print("Must show 'Loading from file' and two files:")
        f_img = loader.load_frames(frames, files, show_file_names=True)
        self.assertEqual(f_img.shape, (4, 200, 200))
        self.assertTrue(np.all(np.equal(f_img, self.frames_1_2_41_42)))

    def test_load_volumes_full(self):
        loader = ImageLoader(self.full_movie)
        # TODO : check all the places for consistency n volumes 1 2 meaning 0 1 actually :(

        frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        volumes = [0] * 10
        volumes.extend([1] * 10)
        files = [self.full_movie] * 20

        v_img = loader.load_volumes(frames, files, volumes)
        self.assertEqual(v_img.shape, (2, 10, 200, 200))
        self.assertTrue(np.all(np.equal(v_img, self.volumes_0_1)))

    def test_load_volumes_half(self):
        loader = ImageLoader(self.full_movie)

        frames = [0, 1, 2, 3, 4,
                  10, 11, 12, 13, 14]
        volumes = [1] * 5
        volumes.extend([2] * 5)
        files = [self.full_movie] * 10

        v_img = loader.load_volumes(frames, files, volumes)
        self.assertEqual(v_img.shape, (2, 5, 200, 200))
        self.assertTrue(np.all(np.equal(v_img, self.half_volumes_0_1)))

        # now let's make sure it breaks when we ask for different number of slices per volume
        volumes = [1] * 6
        volumes.extend([2] * 4)
        files = [self.full_movie] * 10
        with self.assertRaises(AssertionError):
            loader.load_volumes(frames, files, volumes)

if __name__ == "__main__":
    # TODO: test that lists for the db are true int all the time !!!!
    print(TEST_DATA)
    unittest.main()

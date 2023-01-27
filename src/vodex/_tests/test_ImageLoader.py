import numpy as np
from pathlib import Path
import pytest
import tifffile as tif

# import the class to be tested
from vodex import ImageLoader, TiffLoader

# Sample image files to test the loaders with:
# path to folder with test data
TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')
# test data movie, where all the data is in one file
FULL_MOVIE = Path(TEST_DATA, "test_movie.tif")
FULL_MOVIE_FRAMES = [42]
# test data movie, where all the data is split into 3 files
SPLIT_MOVIE = [Path(TEST_DATA, "test_movie", mov) for mov in
               ["mov0.tif", "mov1.tif", "mov2.tif"]]
SPLIT_MOVIE_FRAMES = [7, 18, 17]

# frame characteristics
FRAME_SIZE = (200, 200)

# individual frames
FRAMES = [0, 1, 40, 41]
FRAMES_FILES_FULL_MOVIE = [FULL_MOVIE for _ in range(4)]
FRAMES_TIFF = tif.imread(str(Path(TEST_DATA, 'loader_test', "frames_1_2_41_42.tif")))

# full volumes 0 and 1
VOLUMES_FRAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
VOLUMES_INDICES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
VOLUMES_FILES_FULL_MOVIE = [FULL_MOVIE for _ in range(20)]
VOLUMES_TIFF = tif.imread(str(Path(TEST_DATA, 'loader_test', "volumes_1_2.tif")))

# top half slice of the volumes 0 and 1
SLICES_FRAMES = [0, 1, 2, 3, 4,
                 10, 11, 12, 13, 14]
SLICES_INDICES = [0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1]
SLICES_FILES_FULL_MOVIE = [FULL_MOVIE for _ in range(10)]
SLICES_TIFF = tif.imread(str(Path(TEST_DATA, 'loader_test', "half_volumes_1_2.tif")))

# supported extensions information
SUPPORTED_EXTENSIONS = ['tif', 'tiff']


@pytest.fixture
def image_loader():
    return ImageLoader(FULL_MOVIE)


def test_image_loader_init(image_loader):
    assert isinstance(image_loader, ImageLoader)
    assert image_loader.file_extension == 'tif'
    assert image_loader.supported_extensions == SUPPORTED_EXTENSIONS
    assert isinstance(image_loader.loader, TiffLoader)
    with pytest.raises(AssertionError):
        ImageLoader(Path("dummy.txt"))


def test_image_loader_eq():
    loader1 = ImageLoader(FULL_MOVIE)
    loader2 = ImageLoader(FULL_MOVIE)
    assert loader1 == loader2
    assert loader2 == loader1

    assert loader1.__eq__("ImageLoader") == NotImplemented


def test_image_loader_get_frames_in_file(image_loader):
    assert image_loader.get_frames_in_file(SPLIT_MOVIE[0]) == SPLIT_MOVIE_FRAMES[0]


def test_image_loader_get_frame_size(image_loader):
    assert image_loader.get_frame_size(FULL_MOVIE) == FRAME_SIZE


def test_image_loader_load_frames(image_loader):
    data = image_loader.load_frames(FRAMES, FRAMES_FILES_FULL_MOVIE)
    assert data.shape == (4, 200, 200)
    assert (data == FRAMES_TIFF).all()


def test_image_loader_load_volumes(image_loader):
    # load full volumes
    data = image_loader.load_volumes(VOLUMES_FRAMES, VOLUMES_FILES_FULL_MOVIE, VOLUMES_INDICES)
    assert data.shape == (2, 10, 200, 200)
    assert (data == VOLUMES_TIFF).all()

    # load half volumes
    data = image_loader.load_volumes(SLICES_FRAMES, SLICES_FILES_FULL_MOVIE, SLICES_INDICES)
    assert data.shape == (2, 5, 200, 200)
    assert (data == SLICES_TIFF).all()

    with pytest.raises(AssertionError):
        WRONG_SLICE_INDICES = [0, 0, 0, 0, 0,
                               1, 1, 1, 1, 0]
        image_loader.load_volumes(SLICES_FRAMES, SLICES_FILES_FULL_MOVIE, WRONG_SLICE_INDICES)

from pathlib import Path
import pytest

# import the class to be tested
from vodex import ImageLoader, TiffLoader
from .conftest import (FULL_MOVIE,
                       SPLIT_MOVIE,
                       SPLIT_MOVIE_FRAMES,
                       FRAME_SIZE,
                       FRAMES,
                       FRAMES_1_2_41_42,
                       VOLUMES_FRAMES,
                       VOLUMES_INDICES,
                       VOLUMES_0_1,SLICES_FRAMES,
                       SLICES_INDICES,
                       HALF_VOLUMES_0_1,
                       SUPPORTED_EXTENSIONS)

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
    data = image_loader.load_frames(FRAMES, [FULL_MOVIE for _ in range(4)])
    assert data.shape == (4, 200, 200)
    assert (data == FRAMES_1_2_41_42).all()


def test_image_loader_load_volumes(image_loader):
    # load full volumes
    data = image_loader.load_volumes(VOLUMES_FRAMES, [FULL_MOVIE for _ in range(20)], VOLUMES_INDICES)
    assert data.shape == (2, 10, 200, 200)
    assert (data == VOLUMES_0_1).all()

    # load half volumes
    data = image_loader.load_volumes(SLICES_FRAMES, [FULL_MOVIE for _ in range(10)], SLICES_INDICES)
    assert data.shape == (2, 5, 200, 200)
    assert (data == HALF_VOLUMES_0_1).all()

    with pytest.raises(AssertionError):
        WRONG_SLICE_INDICES = [0, 0, 0, 0, 0,
                               1, 1, 1, 1, 0]
        image_loader.load_volumes(SLICES_FRAMES, [FULL_MOVIE for _ in range(10)], WRONG_SLICE_INDICES)

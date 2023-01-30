"""
Tests for the `vodex.core` module.
"""
import json
from pathlib import Path
import pytest

from vodex import FileManager

TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')
# test data movie, where all the data is split into 3 files
SPLIT_MOVIE_DIR = Path(TEST_DATA, "test_movie")
SPLIT_MOVIE_NAMES = ["mov0.tif", "mov1.tif", "mov2.tif"]
SPLIT_MOVIE = [Path(SPLIT_MOVIE_DIR, mov) for mov in SPLIT_MOVIE_NAMES]
SPLIT_MOVIE_FRAMES = [7, 18, 17]

# test data movie, where all the data is in one file
FULL_MOVIE_DIR = TEST_DATA
FULL_MOVIE_NAMES = ["test_movie.tif"]
FULL_MOVIE = Path(TEST_DATA, FULL_MOVIE_NAMES[0])
FULL_MOVIE_FRAMES = [42]


@pytest.fixture
def file_manager():
    return FileManager(data_dir=SPLIT_MOVIE_DIR, file_type="TIFF")


def test_init_with_data_dir():
    file_manager = FileManager(data_dir=SPLIT_MOVIE_DIR, file_type="TIFF")
    assert file_manager.data_dir == SPLIT_MOVIE_DIR
    assert file_manager.file_names == SPLIT_MOVIE_NAMES
    assert file_manager.num_frames == SPLIT_MOVIE_FRAMES
    assert file_manager.n_files == 3

    file_manager = FileManager(data_dir=FULL_MOVIE_DIR, file_type="TIFF")
    assert file_manager.data_dir == FULL_MOVIE_DIR
    assert file_manager.file_names == FULL_MOVIE_NAMES
    assert file_manager.num_frames == FULL_MOVIE_FRAMES
    assert file_manager.n_files == 1

    with pytest.raises(AssertionError) as e:
        FileManager(data_dir='invalid_dir')
    assert str(e.value) == 'No directory invalid_dir'

    with pytest.raises(AssertionError) as e:
        FileManager(data_dir=FULL_MOVIE_DIR, file_type="AVI")
    assert str(e.value) == 'File type "AVI" is not supported.'


def test_init_with_file_names():
    file_manager = FileManager(data_dir=SPLIT_MOVIE_DIR, file_names=SPLIT_MOVIE_NAMES, file_type="TIFF")
    assert file_manager.data_dir == SPLIT_MOVIE_DIR
    assert file_manager.file_names == SPLIT_MOVIE_NAMES
    assert file_manager.num_frames == SPLIT_MOVIE_FRAMES
    assert file_manager.n_files == 3

    file_manager = FileManager(data_dir=FULL_MOVIE_DIR, file_names=FULL_MOVIE_NAMES, file_type="TIFF")
    assert file_manager.data_dir == FULL_MOVIE_DIR
    assert file_manager.file_names == FULL_MOVIE_NAMES
    assert file_manager.num_frames == FULL_MOVIE_FRAMES
    assert file_manager.n_files == 1

    with pytest.raises(AssertionError) as e:
        FileManager(data_dir=FULL_MOVIE_DIR, file_names=SPLIT_MOVIE_NAMES)

    with pytest.raises(AssertionError) as e:
        FileManager(data_dir=FULL_MOVIE_DIR, file_names=["test_movie.avi", "test_movie.tif"])
    assert str(e.value) == "File_names must be files with the same extension, but got avi, tif"

    with pytest.raises(AssertionError) as e:
        FileManager(data_dir=FULL_MOVIE_DIR, file_names=["test_movie.avi"])
    assert str(e.value) == 'Extension "avi" is not supported.'


def test_init_with_frames_per_file():
    file_manager = FileManager(data_dir=SPLIT_MOVIE_DIR, file_names=SPLIT_MOVIE_NAMES,
                               frames_per_file=[22, 22, 22], file_type="TIFF")
    assert file_manager.data_dir == SPLIT_MOVIE_DIR
    assert file_manager.file_names == SPLIT_MOVIE_NAMES
    assert file_manager.num_frames == [22, 22, 22]
    assert file_manager.n_files == 3

    # frames per file are ignored when no files provided
    file_manager = FileManager(data_dir=SPLIT_MOVIE_DIR,
                               frames_per_file=[22, 22, 22], file_type="TIFF")
    assert file_manager.data_dir == SPLIT_MOVIE_DIR
    assert file_manager.file_names == SPLIT_MOVIE_NAMES
    assert file_manager.num_frames == SPLIT_MOVIE_FRAMES
    assert file_manager.n_files == 3


def test_eq():
    file_m1 = FileManager(SPLIT_MOVIE_DIR)
    file_m2 = FileManager(SPLIT_MOVIE_DIR)
    assert file_m1 == file_m2
    assert file_m2 == file_m1

    file_m3 = FileManager(FULL_MOVIE_DIR)
    assert not (file_m3 == file_m1)

    assert file_m3.__eq__("FileManager") == NotImplemented


def test_str(file_manager):
    result = file_manager.__str__()
    assert result == 'Image files information :\n\n' \
                     'files directory: ' \
                     f'{SPLIT_MOVIE_DIR}\n' \
                     'files [number of frames]: \n' \
                     '0) mov0.tif [7]\n' \
                     '1) mov1.tif [18]\n' \
                     '2) mov2.tif [17]\n'


def test_repr(file_manager):
    result = file_manager.__repr__()
    assert result == 'Image files information :\n\n' \
                     'files directory: ' \
                     f'{SPLIT_MOVIE_DIR}\n' \
                     'files [number of frames]: \n' \
                     '0) mov0.tif [7]\n' \
                     '1) mov1.tif [18]\n' \
                     '2) mov2.tif [17]\n'


def test_find_files(file_manager):
    file_names = file_manager.find_files((".tif",))
    assert file_names == SPLIT_MOVIE_NAMES


def test_check_files(file_manager):
    assert file_manager.check_files(SPLIT_MOVIE_NAMES) == SPLIT_MOVIE_NAMES

    with pytest.raises(AssertionError) as e:
        file_manager.check_files(["test_movie.tif"])
    assert str(e.value) == f'File {Path(SPLIT_MOVIE_DIR, "test_movie.tif")} is not found'


def test_get_frames_per_file(file_manager):
    frames_per_file = file_manager.get_frames_per_file()
    assert frames_per_file == [7, 18, 17]


@pytest.mark.parametrize('new_order', [[2, 1, 0], [0, 2], [1]])
def test_change_files_order(file_manager, new_order):
    file_manager.change_files_order(new_order)
    assert file_manager.data_dir == SPLIT_MOVIE_DIR
    assert file_manager.file_names == [SPLIT_MOVIE_NAMES[i_name] for i_name in new_order]
    assert file_manager.num_frames == [SPLIT_MOVIE_FRAMES[i_name] for i_name in new_order]
    assert file_manager.n_files == len(new_order)


def test_change_files_order_invalid(file_manager):
    with pytest.raises(AssertionError) as e:
        new_order = [2, 1, 0, 0]
        file_manager.change_files_order(new_order)
    assert str(e.value) == "Number of files is smaller than elements in the new order list! "

    with pytest.raises(AssertionError) as e:
        new_order = [2, 0, 0]
        file_manager.change_files_order(new_order)
    assert str(e.value) == "All elements in the new order list must be unique! "

    with pytest.raises(AssertionError) as e:
        new_order = [3, 0]
        file_manager.change_files_order(new_order)
    assert str(e.value) == "All elements in the new order list must be present in the original order: [0, 1, 2]! "

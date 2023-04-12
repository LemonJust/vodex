import numpy as np
import pytest
from pathlib import Path
from typing import List, Tuple, Union
from vodex import Loader


@pytest.fixture
def file_example():
    # Provide a fixture for an example file path
    return "example_file.txt"


# create a class that subclasses Loader:
@pytest.fixture
def concrete_loader(file_example):
    # Provide a fixture for a Loader instance initialized with the example file
    class ConcreteLoader(Loader):
        @staticmethod
        def get_frames_in_file(file: Union[str, Path]) -> int:
            return 5

        @staticmethod
        def get_frame_size(file: Union[str, Path]) -> Tuple[int, int]:
            return (10, 10)

        @staticmethod
        def get_frame_dtype(file: Union[str, Path]) -> np.dtype:
            return np.dtype(np.uint8)

        def load_frames(self, frames: List[int],
                        files: Union[List[str], List[Path]],
                        show_file_names: bool = False,
                        show_progress: bool = True) -> np.ndarray:
            return np.zeros((len(frames), 10, 10))

    return ConcreteLoader(file_example)


def test_frame_size(concrete_loader):
    # Test that the frame_size attribute is set correctly
    assert concrete_loader.frame_size == (10, 10)


def test_data_type(concrete_loader):
    # Test that the data_type attribute is set correctly
    assert concrete_loader.data_type == np.uint8


def test_get_frames_in_file(concrete_loader, file_example):
    # Test the get_frames_in_file method
    assert concrete_loader.get_frames_in_file(file_example) == 5


def test_get_frame_size(concrete_loader, file_example):
    assert concrete_loader.get_frame_size(file_example) == (10, 10)


def test_get_frame_dtype(concrete_loader, file_example):
    assert concrete_loader.get_frame_dtype(file_example) == np.uint8


def test_load_frames(concrete_loader):
    # Test the load_frames method
    frames = [1, 2, 3]
    files = ["file1.txt", "file2.txt", "file3.txt"]

    assert (concrete_loader.load_frames(frames, files) ==
            np.zeros((len(frames), 10, 10))).all()


def test_eq(concrete_loader):
    # Test the __eq__ method
    with pytest.raises(TypeError) as e:
        concrete_loader == concrete_loader
    assert str(e.value) == "__eq__ is Not Implemented for ConcreteLoader and ConcreteLoader"

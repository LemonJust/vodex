"""
This module contains classes to load and collect information from specific file types.

The module contains the following classes:

- `Loader` - A generic loader class. Overwrite the methods to create your own loader.

- `TiffLoader` - A class to work with tiff image files. Subclass of Loader.
Used to get the datatype of the images, get the number of frames in each tiff file and load frames from tiff files.
You can create your own loaders to work with other file types (see Contributions).

"""

import numpy as np
import numpy.typing as npt

from pathlib import Path
from tifffile import TiffFile
from tqdm import tqdm
from typing import Union, Final, Dict, Tuple, List


class Loader:
    """
    A generic loader class.
    Overwrite the methods to create your own loader.

    Args:
        file_example: An example file from the dataset
            to infer the frame size and data type.

    Attributes:
        frame_size: individual frame size (hight, width).
        data_type: datatype.
    """

    def __init__(self, file_example: Union[str, Path]):

        self.frame_size: Tuple[int, int] = self.get_frame_size(file_example)
        self.data_type: np.dtype = self.get_frame_dtype(file_example)

    def __eq__(self, other):
        print(f"__eq__ is Not Implemented for {type(self)} and {type(other)}")

        return NotImplemented

    @staticmethod
    def get_frames_in_file(file: Union[str, Path]) -> int:
        """
        Compute and return the number of frames in a file.

        Args:
            file: the path to the file to get the number of frames for.
        Returns:
            the number of frames in the file.
        """
        raise NotImplementedError

    @staticmethod
    def get_frame_size(file: Union[str, Path]) -> Tuple[int, int]:
        """
        Gets frame size ( height , width ) from a tiff file.

        Args:
            file: the path to the file to get the size of the frame for.
        Returns:
            ( height , width ) height and width of an individual frame in pixels.
        """
        raise NotImplementedError

    @staticmethod
    def get_frame_dtype(file: Union[str, Path]) -> np.dtype:
        """
        Gets the datatype of the frame.

        Args:
            file: the path to the file to get the datatype of the frame for.
        Returns:
            datatype of the frame.
        """
        raise NotImplementedError

    def load_frames(self, frames: List[int], files: Union[List[str], List[Path]],
                    show_file_names: bool = False, show_progress: bool = True) -> npt.NDArray:
        """
        Load frames from files and return as an array (frame, y, x).

        Args:
            frames: list of frames inside corresponding files to load
            files: list of files corresponding to each frame
            show_file_names: whether to print the file from which the frames are loaded on the screen.
            show_progress: whether to show the progress bar of how many frames have been loaded.
        Returns:
            3D array of requested frames (frame, y, x)
        """
        raise NotImplementedError


class TiffLoader(Loader):
    """
    A class to work with tiff image files.
    It is used to get the datatype of the images, get the number
    of frames in each tiff file and load frames from tiff files.
    You can create your own loaders to work with other file types.

    Args:
        file_example: An example tif file from the dataset
            to infer the frame size and data type.

    Attributes:
        frame_size: individual frame size (hight, width).
        data_type: datatype.
    """

    def __init__(self, file_example: Union[str, Path]):
        super().__init__(file_example)

        self.frame_size: Tuple[int, int] = self.get_frame_size(file_example)
        self.data_type: np.dtype = self.get_frame_dtype(file_example)

    def __eq__(self, other):
        if isinstance(other, TiffLoader):
            same_fs = self.frame_size == other.frame_size
            same_dt = self.data_type == other.data_type
            return same_fs and same_dt

        else:
            print(f"__eq__ is Not Implemented for {TiffLoader} and {type(other)}")
            return NotImplemented

    @staticmethod
    def get_frames_in_file(file: Union[str, Path]) -> int:
        """
        Compute and return the number of frames in a file.

        Args:
            file: the name of a file relative to data_dir to get the number of frames for.
        Returns:
            the number of frames in the file.
        """
        # TODO : try-catch here ?
        # setting multifile to false since sometimes there is a problem with the corrupted metadata
        # not using metadata, since for some files it is corrupted for unknown reason ...
        stack = TiffFile(file, _multifile=False)
        n_frames = len(stack.pages)
        stack.close()

        return n_frames

    @staticmethod
    def get_frame_size(file: Union[str, Path]) -> Tuple[int, int]:
        """
        Gets frame size ( height , width ) from a tiff file.

        Args:
            file: the path to the file to get the size of the frame for.
        Returns:
            ( height , width ) height and width of an individual frame in pixels.
        """
        # TODO : try-catch here ?
        # setting multifile to false since sometimes there is a problem with the corrupted metadata
        # not using metadate, since for some files it is corrupted for unknown reason ...
        stack = TiffFile(file, _multifile=False)
        page = stack.pages.get(0)
        h, w = page.shape
        stack.close()
        return h, w

    @staticmethod
    def get_frame_dtype(file: Union[str, Path]) -> np.dtype:
        """
        Gets the datatype of the frame.

        Args:
            file: the path to the file to get the datatype of the frame for.
        Returns:
            datatype of the frame.
        """
        # TODO : try-catch here ?
        # setting multifile to false since sometimes there is a problem with the corrupted metadata
        # not using metadate, since for some files it is corrupted for unknown reason ...
        stack = TiffFile(file, _multifile=False)
        page = stack.pages.get(0)
        data_type = page.dtype
        stack.close()
        return data_type

    def load_frames(self, frames: List[int], files: Union[List[str], List[Path]],
                    show_file_names: bool = False, show_progress: bool = True) -> npt.NDArray:
        """
        Load frames from files and return as an array (frame, y, x).

        Args:
            frames: list of frames inside corresponding files to load
            files: list of files corresponding to each frame
            show_file_names: whether to print the file from which the frames are loaded on the screen.
            show_progress: whether to show the progress bar of how many frames have been loaded.
        Returns:
            3D array of requested frames (frame, y, x)
        """

        def print_file_name():
            if show_file_names:
                print(f'Loading from file:\n {tif_file}')

        if show_file_names:
            # Setting show_progress to False, show_progress can't be True when show_file_names is True
            if show_progress:
                show_progress = False
        hide_progress = not show_progress

        # prepare an empty array:
        h, w = self.frame_size
        img = np.zeros((len(frames), h, w), dtype=self.data_type)

        # initialise tif file and open the stack
        tif_file = files[0]
        stack = TiffFile(tif_file, _multifile=False)

        print_file_name()
        for i, frame in enumerate(tqdm(frames, disable=hide_progress, unit='frames')):
            # check if the frame belongs to an opened file
            if files[i] != tif_file:
                # switch to a different file
                tif_file = files[i]
                stack.close()
                print_file_name()
                stack = TiffFile(tif_file, _multifile=False)
            img[i, :, :] = stack.asarray(frame)
        stack.close()
        return img

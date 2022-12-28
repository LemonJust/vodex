"""
This module contains classes to load and collect information from the imaging data.

The module contains the following classes:

- `TiffLoader` - A class to work with tiff image files.
It is used to get the datatype of the images, get the number of frames in each tiff file and load frames from tiff files.
You can create your own loaders to work with other file types.

- `ImageLoader` - Chooses appropriate loader based on the type of the imaging files,
collects information about the datatype, number of frames per file and loads data from files.
"""

import numpy as np
from pathlib import Path
from tifffile import TiffFile
from tqdm import tqdm
from typing import Union, Final, Dict


class TiffLoader:
    """
    A class to work with tiff image files.
    It is used to get the datatype of the images, get the number
    of frames in each tiff file and load frames from tiff files.
    You can create your own loaders to work with other file types.

    Args:
        file_example: An example file file from the dataset
            to infer the frame size and data type.

    Attributes:
        frame_size (tuple[int,int]): individual frame size (hight, width).
        data_type (np.dtype): datatype.
    """

    def __init__(self, file_example: Union[str, Path]):

        self.frame_size = self.get_frame_size(file_example)
        self.data_type = self.get_frame_dtype(file_example)

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
    def get_frame_size(file: Union[str, Path]) -> tuple[int, int]:
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

    def load_frames(self, frames: list[int], files: Union[list[str], list[Path]],
                    show_file_names: bool = False, show_progress: bool = True) -> np.ndarray:
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


class ImageLoader:
    """
    Chooses appropriate loader based on the type of the imaging files,
    collects information about the datatype, number of frames per file and loads data from files.

    Args:
        file_example : the path to a file example (one file from the whole dataaset).
            needed to get file type and initialise the corresponding data loader.

    Attributes:
        loader_map: a dictionary that maps the file extensions to their loaders.
        supported_extension (list[str]): list of all the supported file extensions.
        file_extension (str): the file extension of the provided file example.
        loader: a loader class initialised using the file example.

    """

    def __init__(self, file_example: Union[str, Path]):

        # A list of supported formats. Create mapping from file formats to loaders.
        # (Currently only supports files with tif extensions)
        # Expand this dictionary to add support for other file formats.
        # 1. *.tif and *.tiff support is implemented using tifffile package
        # 2. ...
        self.loader_map: Dict[str, type] = {".tif": TiffLoader,
                                            ".tiff": TiffLoader}
        self.supported_extension = self.loader_map.keys()

        # Right now everything we need to initialise a TiffLoader we can get from one example file:
        # file extension and the size and data type of one frame.
        # If you are adding a custom loader and it needs access to the whole recording
        # or manual input for initialisation or a config file ... or anything else,
        # you need to change the initialisation of the ImageClass as well.
        self.file_extension = file_example.suffix
        assert self.file_extension in self.supported_extension, \
            f"Only files with the following extensions are supported: {self.supported_extension}, but" \
            f"{self.file_extension} was given"

        # Pick the loader and initialise it with the data directory:
        # chooses the proper loader based on the files extension.
        # Add your class to name to the supported variable types when adding support to other file formats.
        self.loader: Union[TiffFile, ] = self.loader_map[self.file_extension](file_example)

    def __eq__(self, other):
        if isinstance(other, ImageLoader):
            is_same = [
                self.supported_extension == other.supported_extension,
                self.file_extension == other.file_extension,
                self.loader == other.loader
            ]
            return np.all(is_same)
        else:
            print(f"__eq__ is Not Implemented for {ImageLoader} and {type(other)}")
            return NotImplemented

    def get_frames_in_file(self, file_name: Union[str, Path]) -> int:
        """
        Calculates and returns the number of frames in a given file.

        Args:
            file_name: the name of the file to get the number of frames for.

        Returns:
            the number of frames in the file.
        """
        return self.loader.get_frames_in_file(file_name)

    def get_frame_size(self, file_name: Union[str, Path]) -> tuple[int, int]:
        """
        Gets frame size ( height , width ) from an image files.

        Args:
            file_name: the path to the file to get the size of the frame for.
        Returns:
            ( height , width ) height and width of an individual frame in pixels.
        """
        return self.loader.get_frame_size(file_name)

    def load_frames(self, frames: list[int], files: Union[list[str], list[Path]],
                    show_file_names: bool = False, show_progress: bool = True) -> np.ndarray:
        """
        Loads specified frames from specified files.

        Args:
            frames: list of frames IN FILES to load.
            files: a file for every frame
            show_file_names: whether to print the names of the files from which the frames are loaded.
                Setting it to True will turn off show_progress.
            show_progress: whether to show the progress bar of how many frames have been loaded.
                Won't have effect of show_file_names is True.
        Returns:
            3D array of shape (n_frames, height, width)
        """
        return self.loader.load_frames(frames, files,
                                       show_file_names=show_file_names,
                                       show_progress=show_progress)

    def load_volumes(self,
                     frame_in_file: list[int],
                     files: Union[list[str], list[Path]],
                     volumes: list[int],
                     show_file_names: bool = False, show_progress: bool = True) -> np.ndarray:
        """
        Loads specified frames from specified files and shapes them into volumes.

        Args:
            frame_in_file: list of frames IN FILES to load
                (relative to the beginning of the file from which you are loading).
            files: a file for every frame
            volumes: a volume for every frame where that frame belongs
            show_file_names: whether to print the names of the files from which the frames are loaded.
                                Setting it to True will turn off show_progress.
            show_progress: whether to show the progress bar of how many frames have been loaded.
                Won't have effect of show_file_names is True.
        Returns:
            4D array of shape (number of volumes, zslices, height, width)
        """
        # TODO : do I need to check anything else here???
        #  Like that the frames are in increasing order per file
        #  ( maybe not here but in the experiment ,
        #       before we turn them into frames_in_file )
        # get frames and info
        frames = self.loader.load_frames(frame_in_file, files,
                                         show_file_names=show_file_names,
                                         show_progress=show_progress)
        n_frames, w, h = frames.shape

        # get volume information
        i_volume, count = np.unique(volumes, return_counts=True)
        # you can use this method to load portions of the volumes, so fpv will be smaller
        n_volumes, fpv = len(i_volume), count[0]
        assert np.all(count == fpv), "Can't have different number of frames per volume!"

        frames = frames.reshape((n_volumes, fpv, w, h))
        return frames

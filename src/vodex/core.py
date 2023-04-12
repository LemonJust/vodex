"""
This module provides the core classes for loading and preprocessing imaging data.

'ImageLoader' is the core class that loads the image data and selects the appropriate loader based on the file type. It
also collects information about the data type and number of frames per file.

'FileManager', 'FrameManager', and 'VolumeManager' are core classes that preprocess information about the experiment
data. The 'FileManager' class contains information about the location and file type of the image files,
while the 'FrameManager' class contains information about the number of frames in the experiment and the mapping of
frames to files. The 'VolumeManager' class contains information about the image volumes in the experiment,
including the number of frames per volume and the mapping of frames to volumes.
"""
from pathlib import Path
from typing import Union, List, Tuple, Dict

import numpy as np
import numpy.typing as npt

from .loaders import Loader, TiffLoader


# This section of the code is related to adding support for new file types. ___________________________________________

# 'VX_SUPPORTED_TYPES' is a dictionary that lists the supported file formats and their corresponding extensions.
# Currently, only files in TIFF format are supported using the tifffile package.
# To add support for other file formats, additional entries should be added to this dictionary in the form of
# 'FileType': ('.extension') or 'FileType': ('.extension1', '.extension2', '.extension3').
VX_SUPPORTED_TYPES: Dict[str, tuple] = {'TIFF': ('.tif', '.tiff')}

# 'VX_EXTENSION_TO_TYPE' is a dictionary that maps file extensions to their corresponding file formats.
# For example, it maps the '.tif' and '.tiff' extensions to the 'TIFF' file type.
# To add support for other file formats, additional entries should be added to this dictionary in the form of
# 'extension': 'FileType' or 'extension1': 'FileType', 'extension2': 'FileType', 'extension3': 'FileType'.
VX_EXTENSION_TO_TYPE: Dict[str, str] = {'tif': 'TIFF', 'tiff': 'TIFF'}

# 'VX_EXTENSION_TO_LOADER' is a dictionary that maps file extensions to the corresponding loader classes. For example,
# it maps the '.tif' and '.tiff' extensions to the TiffLoader class.
# To add support for other file formats, additional entries should be added to this dictionary in the form of
# 'extension': 'LoaderClass' or 'extension1': 'LoaderClass', 'extension2': 'LoaderClass', 'extension3': 'LoaderClass'.
VX_EXTENSION_TO_LOADER: Dict[str, type] = {'tif': TiffLoader, 'tiff': TiffLoader}


# _____________________________________________________________________________________________________________________


class ImageLoader:
    """
    The 'ImageLoader' class is responsible for choosing the appropriate loader for a given imaging file, collecting
    information about the data type, number of frames per file, and loading data from files.

    Args:
        file_example : The path to a file example (one file from the whole dataset).
                    This is used to determine the file type and initialize the corresponding data loader.

    Attributes:
        supported_extensions: A list of all the supported file extensions.
        file_extension: The file extension of the provided file example.
        loader:  The loader class initialized using the file example.

    """

    def __init__(self, file_example: Path):
        """
        Initializes the ImageLoader class by determining the file extension, checking that it is a supported format,
        and initializing the appropriate loader class.
        """

        self.supported_extensions: List[str] = list(VX_EXTENSION_TO_LOADER.keys())
        # suffix has the dot at the beginning, need to strip
        self.file_extension: str = file_example.suffix.lstrip('.')
        assert self.file_extension in self.supported_extensions, \
            f"Only files with the following extensions are supported: {self.supported_extensions}, but" \
            f"{self.file_extension} was given"

        # Pick the loader and initialise it with the data directory:
        # chooses the proper loader based on the file extension.
        # Add your class to VX_EXTENSION_TO_LOADER when adding support to other file formats.
        self.loader: Loader = VX_EXTENSION_TO_LOADER[self.file_extension](file_example)

    def __eq__(self, other):
        """
        Compares two ImageLoader instances to see if they are equal.
        """
        if isinstance(other, ImageLoader):
            is_same = [
                self.supported_extensions == other.supported_extensions,
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

    def get_frame_size(self, file_name: Union[str, Path]) -> Tuple[int, int]:
        """
        Returns the size of an individual frame in pixels (height and width).

        Args:
            file_name: the path to the file to get the size of the frame for.
        Returns:
            ( height , width ) height and width of an individual frame in pixels.
        """
        return self.loader.get_frame_size(file_name)

    def load_frames(self, frames: List[int], files: Union[List[str], List[Path]],
                    show_file_names: bool = False, show_progress: bool = True) -> npt.NDArray:
        """
         Loads specified frames from specified files, and returns as a 3D array of shape (n_frames, height, width).

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
                     frame_in_file: List[int],
                     files: Union[List[str], List[Path]],
                     volumes: List[int],
                     show_file_names: bool = False, show_progress: bool = True) -> npt.NDArray:
        """
         Loads specific volumes of data, where a volume is defined as a set of frames.
         This method returns a 4D array of shape (n_volumes, n_frames_per_volume, height, width).

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
        # get frames and info
        frames = self.loader.load_frames(frame_in_file, files,
                                         show_file_names=show_file_names,
                                         show_progress=show_progress)
        n_frames, w, h = frames.shape

        # get volume information
        i_volume, count = np.unique(volumes, return_counts=True)
        # you can use this method to load portions of the volumes (slices), so fpv will be smaller than a full volume
        n_volumes, fpv = len(i_volume), count[0]
        assert np.all(count == fpv), "Can't have different number of frames per volume!"

        frames = frames.reshape((n_volumes, fpv, w, h))
        return frames


class FileManager:
    """
    The 'FileManager' class is used to collect and store information about image files, including their location,
    file type, and number of frames per file. It can either search for all files with a specific file extension in a
    provided data directory (order them alphabetically), or use a provided list of file names (in the provided order).
    The class initializes an 'ImageLoader' to calculate the number of frames per file if it is not provided.

    The class raises an error if the data directory does not exist, no files of the specified file type are found,
    or if the number of frames per file is not provided or is not a list of integers.

    Args:
        data_dir: path to the folder with the files, ends with "/" or "\\"
        file_names: names of files relative to the data_dir
        frames_per_file: number of frames in each file. Will be used ONLY if the file_names were provided.
        file_type: file type to search for (if files are not provided). Must be a key in the VX_SUPPORTED_TYPES dict.

    Attributes:
        data_dir: the directory with all the imaging data
        file_names: names of files relative to the data_dir
        num_frames: a number of frames per file
        n_files: total number of image files
    """

    def __init__(self, data_dir: Union[str, Path], file_type: str = "TIFF",
                 file_names: List[str] = None, frames_per_file: List[int] = None):

        # 1. get data_dir and check it exists
        self.data_dir: Path = Path(data_dir)
        assert self.data_dir.is_dir(), f"No directory {self.data_dir}"

        # 2. get files and file type
        if file_names is not None:
            tags = [name.split(".")[-1] for name in file_names]
            # check that all the elements of the list are same
            assert len(set(tags)) == 1, f"File_names must be files with the same extension, " \
                                        f"but got {', '.join(sorted(set(tags)))}"
            assert tags[0] in VX_EXTENSION_TO_TYPE, f'Extension "{tags[0]}" is not supported.'
            file_type = VX_EXTENSION_TO_TYPE[tags[0]]

        self.file_type = file_type
        assert self.file_type in VX_SUPPORTED_TYPES, f'File type "{self.file_type}" is not supported.'
        file_extensions = VX_SUPPORTED_TYPES[self.file_type]

        self.num_frames = None
        # TODO : check in accordance with the file extension/ figure out file type from files when provided
        if file_names is None:
            # if files are not provided , search for tiffs in the data_dir
            self.file_names: List[str] = self.find_files(file_extensions)
        else:
            # if a list of files is provided, check it's in the folder
            self.file_names: List[str] = self.check_files(file_names)
            if frames_per_file is not None:
                # not recommended! this information is taken as is and is not verified...
                self.num_frames: List[int] = frames_per_file

        assert len(self.file_names) > 0, f"No files of type {file_type} [extensions {file_extensions}]\n" \
                                         f" in {data_dir}"

        # 3. Get number of frames per file (if it wasn't entered manually)
        if self.num_frames is None:
            # if number of frames not provided , search for tiffs in the data_dir
            self.num_frames: List[int] = self.get_frames_per_file()

        self.n_files: int = len(self.file_names)

    def __eq__(self, other):
        """
        Compares two FileManager instances to see if they are equal.
        """
        if isinstance(other, FileManager):
            is_same = [
                self.data_dir == other.data_dir,
                self.file_names == other.file_names,
                self.num_frames == other.num_frames,
                self.n_files == other.n_files
            ]

            return np.all(is_same)
        else:
            print(f"__eq__ is Not Implemented for {FileManager} and {type(other)}")
            return NotImplemented

    def __str__(self):
        description = f"Image files information :\n\n"
        description = description + f"files directory: {self.data_dir}\n"
        description = description + f"files [number of frames]: \n"
        for (i_file, (file_name, num_frames)) in enumerate(zip(self.file_names, self.num_frames)):
            description = description + f"{i_file}) {file_name} [{num_frames}]\n"
        return description

    def __repr__(self):
        return self.__str__()

    def find_files(self, file_extensions: Tuple[str]) -> List[str]:
        """
        Searches for files ending with the provided file extension in the data directory.
        Sorts the names alphabetically in ascending order (from A to Z),
        sorting is case-insensitive (upper case letters are NOT prioritized).

        Args:
            file_extensions: extensions of files to search for
        Returns:
            A sorted list of file names. File names are with the extension, relative to the data directory
            (names only, not full paths to files)
        """
        files = (p.resolve() for p in Path(self.data_dir).glob('*') if p.suffix in file_extensions)
        file_names = [file.name for file in files]
        file_names.sort(key=str.lower)
        return file_names

    def check_files(self, file_names: List[str]) -> List[str]:
        """
        Given a list of files checks that files are in the data directory.
        Throws an error if any of the files are missing.

        Args:
            file_names: list of filenames to check.

        Returns:
            If the files are all present in the directory, returns the file_names.
        """
        # TODO: List all the missing files, not just the first encountered.
        files = [self.data_dir.joinpath(file) for file in file_names]
        for file in files:
            assert file.is_file(), f"File {file} is not found"
        return file_names

    def get_frames_per_file(self) -> List[int]:
        """
        Get the number of frames per file.

        Returns:
            a list with number of frames per file.
        """
        frames_per_file = []
        #    Initialise ImageLoader:
        #    will pick the image loader that works with the provided file type
        loader: ImageLoader = ImageLoader(self.data_dir.joinpath(self.file_names[0]))
        for file in self.file_names:
            n_frames = loader.get_frames_in_file(self.data_dir.joinpath(file))
            frames_per_file.append(n_frames)
        return frames_per_file

    def change_files_order(self, order: List[int]) -> None:
        """
        Changes the order of the files. If you notice that files are in the wrong order, provide the new order.
        If you wish to exclude any files, get rid of them ( don't include their IDs into the new order ).

        Args:
            order: The new order in which the files follow. Refer to file by it's position in the original list.
                    Should be the same length as the number of files in the original list or smaller, no duplicates.
        """
        assert len(order) <= self.n_files, \
            "Number of files is smaller than elements in the new order list! "
        assert len(order) == len(set(order)), \
            "All elements in the new order list must be unique! "
        assert set(order).issubset(list(range(self.n_files))), \
            f"All elements in the new order list must be present in the original order: {list(range(self.n_files))}! "

        self.file_names = [self.file_names[i] for i in order]
        self.num_frames = [self.num_frames[i] for i in order]
        self.n_files = len(self.file_names)


class FrameManager:
    """
    A class containing information about the image frames in the experiment:
    total number of frames, and mapping of the frames to files.

    Args:
        file_manager: FileManager with the information about the files.

    Attributes:
        file_manager: FileManager with the information about the files.
        n_frames: total number of frames in the experiment (global frames).
        frame_to_file: a mapping for each global frame to a file where the frame is stored
        frame_in_file: a mapping for each global frame to a frame number relative to the beginning of the
                        corresponding file
    """

    def __init__(self, file_manager: FileManager):
        self.file_manager: FileManager = file_manager
        self.n_frames: int = int(np.sum(self.file_manager.num_frames))
        self.frame_to_file: List[int]
        self.frame_in_file: List[int]
        self.frame_to_file, self.frame_in_file = self._get_frame_mapping()

    def __eq__(self, other):
        if isinstance(other, FrameManager):
            is_same = [
                self.file_manager == other.file_manager,
                self.frame_to_file == other.frame_to_file,
                self.frame_in_file == other.frame_in_file
            ]

            return np.all(is_same)
        else:
            print(f"__eq__ is Not Implemented for {FrameManager} and {type(other)}")
            return NotImplemented

    @classmethod
    def from_dir(cls, data_dir: Union[Path, str], file_type: str = "TIFF",
                 file_names: List[str] = None, frames_per_file: List[int] = None):
        """
        Create a FrameManager object from files in directory.

        Args:
            data_dir: path to the folder with the files, ends with "/" or "\\"
            file_type: file type to search for (if files are not provided). Must be a key in the VX_SUPPORTED_TYPES dict.
            file_names: names of files relative to the data_dir
            frames_per_file: number of frames in each file. Will be used ONLY if the file_names were provided.

        Returns:
            (FileManager): Initialised FileManager object.
        """
        file_manager = FileManager(data_dir, file_type=file_type,
                                   file_names=file_names, frames_per_file=frames_per_file)
        return cls(file_manager)

    def _get_frame_mapping(self) -> (List[int], List[int]):
        """
        Calculates frame range in each file and returns a file index for each frame and frame index in the file.
        Used to figure out in which stack the requested frames is.
        Frame number starts at 0.

        Returns:
            Two lists mapping frames to files. 'frame_to_file' is a list of length equal to the total number of
            frames in all the files, where each element corresponds to a frame and contains the file index,
            of the file where that frame can be found. 'in_file_frame' is a list of length equal to the total number of
            frames in all the files, where each element corresponds to the index of the frame inside the file.
        """
        frame_to_file = []
        frame_in_file = []

        for file_idx in range(self.file_manager.n_files):
            n_frames = self.file_manager.num_frames[file_idx]
            frame_to_file.extend(n_frames * [file_idx])
            frame_in_file.extend(range(n_frames))

        return frame_to_file, frame_in_file

    def __str__(self):
        return f"Total {np.sum(self.file_manager.num_frames)} frames."

    def __repr__(self):
        return self.__str__()


class VolumeManager:
    """
    A class containing information about the image volumes in the experiment: frames per volume,
    number of full volumes, and mapping of the frames to volumes.

    Args:
        fpv: frames per volume, number of frames in one volume
        fgf: first good frame, the first frame in the imaging session that is at the top of a volume.
            For example if you started imaging at the top of the volume, fgf = 0,
            but if you started somewhere in the middle, the first good frame is , for example, 23 ...
        frame_manager: FrameManager object with the information about the frames in the experiment.

    Attributes:
        fpv: frames per volume, number of frames in one volume
        frame_manager: FrameManager object with the information about the frames in the experiment.
        file_manager: FileManager object with the information about the files.
        n_frames: total number of frames in the experiment
        n_head: (same as fgf) number of frames at the beginning of the recording,
            that do not correspond to a full volume. If the recording starts at the top of a volume, it will be 0.
        n_tail: number of frames at the end of the recording, that do not correspond to a full volume.
        full_volumes: number of full volumes in the recording.
        frame_to_z: mapping of global frames to a z-slice (a slice relative to the top of the volume)
        frame_to_vol: mapping of a global frame to a full volume: -1 for head ( not full volume at the beginning )
                volume number for full volumes : 0, 1, ,2 3, ..., -2 for tail (not full volume at the end )
    """

    def __init__(self, fpv: int, frame_manager: FrameManager, fgf: int = 0):

        assert isinstance(fpv, int), "fpv must be an integer"
        assert isinstance(fgf, int), "fgf must be an integer"

        # frames per volume
        self.fpv: int = int(fpv)

        # get total number of frames
        self.frame_manager: FrameManager = frame_manager
        self.file_manager: FileManager = frame_manager.file_manager
        self.n_frames: int = int(np.sum(self.file_manager.num_frames))

        # prepare info about frames at the beginning, full volumes and frames at the end
        # first good frame, start counting from 0 : 0, 1, 2, 3, ...
        # n_head is the number of frames before the first frame of the first full volume
        # n_tail is the number of frames after the last frame of the last full volume
        self.n_head: int = int(fgf)
        full_volumes, n_tail = divmod((self.n_frames - self.n_head), self.fpv)
        self.full_volumes: int = int(full_volumes)
        self.n_tail: int = int(n_tail)

        # map frames to slices an full volumes:
        self.frame_to_z: List[int] = self._get_frames_to_z_mapping()
        self.frame_to_vol: List[int] = self._get_frames_to_volumes_mapping()

    def __eq__(self, other):
        if isinstance(other, VolumeManager):
            is_same = [
                self.fpv == other.fpv,
                self.frame_manager == other.frame_manager,
                self.file_manager == other.file_manager,
                self.n_frames == other.n_frames,
                self.n_head == other.n_head,
                self.full_volumes == other.full_volumes,
                self.n_tail == other.n_tail,
                self.frame_to_z == other.frame_to_z,
                self.frame_to_vol == other.frame_to_vol
            ]

            return np.all(is_same)
        else:
            print(f"__eq__ is Not Implemented for {VolumeManager} and {type(other)}")
            return NotImplemented

    def _get_frames_to_z_mapping(self) -> List[int]:
        """
        maps frames to z-slices
        """
        z_per_frame_list = np.arange(self.fpv).astype(int)
        # set at what z the imaging starts and ends
        i_from = self.fpv - self.n_head
        i_to = self.n_tail - self.fpv
        # map frames to z
        frame_to_z = np.tile(z_per_frame_list, self.full_volumes + 2)[i_from:i_to]
        return frame_to_z.tolist()

    def _get_frames_to_volumes_mapping(self) -> List[int]:
        """
        maps frames to volumes
        -1 for head ( not full volume at the beginning )
        volume number for full volumes : 0, 1, ,2 3, ...
        -2 for tail (not full volume at the end )
        """
        # TODO : make sure n_head is not larger than full volume?
        frame_to_vol = [-1] * self.n_head
        for vol in np.arange(self.full_volumes):
            frame_to_vol.extend([int(vol)] * self.fpv)
        frame_to_vol.extend([-2] * self.n_tail)
        return frame_to_vol

    def __str__(self):
        description = ""
        description = description + f"Total frames : {self.n_frames}\n"
        description = description + f"Volumes start on frame : {self.n_head}\n"
        description = description + f"Total good volumes : {self.full_volumes}\n"
        description = description + f"Frames per volume : {self.fpv}\n"
        description = description + f"Tailing frames (not a full volume , at the end) : {self.n_tail}\n"
        return description

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_dir(cls, data_dir: Union[str, Path], fpv: int, fgf: int = 0, file_type: str = 'TIFF',
                 file_names: List[str] = None, frames_per_file: List[int] = None):
        """
        Creates a VolumeManager object from directory.

        Args:
            data_dir: path to the folder with the files, ends with "/" or "\\"
            file_type: file type to search for (if files are not provided). Must be a key in the VX_SUPPORTED_TYPES dict.
            fpv: frames per volume, number of frames in one volume
            fgf: first good frame, the first frame in the imaging session that is at the top of a volume.
                For example if you started imaging at the top of the volume, fgf = 0,
                but if you started somewhere in the middle, the first good frame is , for example, 23 ...
            file_names: names of files relative to the data_dir
            frames_per_file: number of frames in each file. Will be used ONLY if the file_names were provided.

        Returns:
            (VolumeManager): Initialised VolumeManager object.
        """
        file_manager = FileManager(data_dir, file_type=file_type, file_names=file_names,
                                   frames_per_file=frames_per_file)
        frame_manager = FrameManager(file_manager)
        return cls(fpv, frame_manager, fgf=fgf)



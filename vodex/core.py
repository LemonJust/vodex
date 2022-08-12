"""
Classes to specify the experimental conditions and load necessary data.
"""
from tifffile import TiffFile
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from dbmethods import DbWriter, DbReader


class TiffLoader:
    """
    A class to work with tiff images.
    """

    def __init__(self, file_example):
        """
        :param file_example: an example file file from the data
        :type file_example: Union(str, Path)
        """
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
    def get_frames_in_file(file):
        """
        returns the number of frames in file
        file_name: name of file relative to data_dir
        """
        # TODO : try-catch here ?
        # setting multifile to false since sometimes there is a problem with the corrupted metadata
        # not using metadate, since for some files it is corrupted for unknown reason ...
        stack = TiffFile(file, _multifile=False)
        n_frames = len(stack.pages)
        stack.close()

        return n_frames

    @staticmethod
    def get_frame_size(file):
        """
        Gets frame size ( height , width ).

        :return: height and width of an individual frame in pixels
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
    def get_frame_dtype(file):
        """
        Gets frame datatype

        :return: datatype of the frame
        """
        # TODO : try-catch here ?
        # setting multifile to false since sometimes there is a problem with the corrupted metadata
        # not using metadate, since for some files it is corrupted for unknown reason ...
        stack = TiffFile(file, _multifile=False)
        page = stack.pages.get(0)
        data_type = page.dtype
        stack.close()
        return data_type

    def load_frames(self, frames, files, show_file_names=False, show_progress=True):
        """
        Load frames from files and return as an array (frame, y, x).

        :param frames: list of frames inside corresponding files to load
        :type frames: list[int]

        :param files: list of files corresponding to each frame
        :type files: Union(list[str],list[Path])

        :param show_file_names: whether to print the file from which the frames are loaded on the screen.
        :type show_file_names: bool

        :param show_progress: whether to show the progress bar of how many frames have been loaded.
        :type show_progress: bool

        :return: 3D array of requested frames (frame, y, x)
        :rtype: numpy.ndarray
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
        for i, frame in enumerate(tqdm(frames, disable=hide_progress)):
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
    Loads Images. Deals with different types of Images
    """

    def __init__(self, file_example):
        """
        file_example : needed to get file extention and get the frame size
        """

        self.supported_extension = [".tif", ".tiff"]
        self.file_extension = file_example.suffix
        assert self.file_extension in self.supported_extension, \
            f"Only files with the following extensions are supported: {self.supported_extension}, but" \
            f"{self.file_extension} was given"
        # pick the loader and initialise it with the data directory
        self.loader = self.choose_loader(file_example)

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

    def choose_loader(self, file_example):
        """
        Chooses the proper loader based on the files extension.
        """
        if self.file_extension == ".tif" or self.file_extension == ".tiff":
            return TiffLoader(file_example)

    def get_frames_in_file(self, file_name):
        return self.loader.get_frames_in_file(file_name)

    def get_frame_size(self, file_name):
        return self.loader.get_frame_size(file_name)

    def load_frames(self, frames, files, show_file_names=False, show_progress=True):
        """
        Loads specified frames from specified files.
        :param frames: list of frames IN FILES to load.
        :type frames: list[int]
        :param files: a file for every frame
        :type files: Union(list[str],list[Path])

        :param show_file_names: whether to print the names of the files from which the frames are loaded.
                                Setting it to True will turn off show_progress.
        :type show_file_names: bool
        :param show_progress: whether to show the progress bar of how many frames have been loaded.
        Won't have effect of show_file_names is True.
        :type show_progress: bool

        :return: 3D array of shape (n_frames, height, width)
        :rtype: numpy.ndarray
        """
        return self.loader.load_frames(frames, files,
                                       show_file_names=show_file_names,
                                       show_progress=show_progress)

    def load_volumes(self, frame_in_file, files, volumes, show_file_names=False, show_progress=True):
        """
        Loads specified frames from specified files and shapes them into volumes.
        :param frame_in_file: list of frames IN FILES to load.
        :type frame_in_file: list[int]
        :param files: a file for every frame
        :type files: Union(list[str],list[Path])
        :param volumes: a volume for every frame
        :type volumes: list[int]

        :param show_file_names: whether to print the names of the files from which the frames are loaded.
                                Setting it to True will turn off show_progress.
        :type show_file_names: bool
        :param show_progress: whether to show the progress bar of how many frames have been loaded.
        Won't have effect of show_file_names is True.
        :type show_progress: bool

        :return: 4D array of shape (number of volumes, zslices, height, width)
        :rtype: numpy.ndarray
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


class FileManager:
    """
    Figures out stuff concerning the many files. For example in what order do stacks go?
    Will grab all the files with the provided file_extension
    in the provided folder and order them alphabetically.
    """

    def __init__(self, data_dir, file_names=None, frames_per_file=None, file_extension=".tif"):
        """
        :param data_dir: path to the folder with the files, ends with "/" or "\\"
        :type data_dir: Union(str,Path)
        """
        # 1. get data_dir and check it exists
        self.data_dir = Path(data_dir)
        assert self.data_dir.is_dir(), f"No directory {self.data_dir}"

        # 2. get files
        if file_names is None:
            # if files are not provided , search for tiffs in the data_dir
            self.file_names = self.find_files(file_extension)
        else:
            # if a list of files is provided, check it's in the folder
            self.file_names = self.check_files(file_names)
        # 3. Initialise ImageLoader
        #    will pick the image loader that works with the provided file type
        self.loader = ImageLoader(self.data_dir.joinpath(self.file_names[0]))

        # 4. Get number of frames per file
        if frames_per_file is None:
            # if number of frames not provided , search for tiffs in the data_dir
            self.num_frames = self.get_frames_per_file()
        else:
            # if provided ... we'll trust you - hope these numbers are correct
            self.num_frames = frames_per_file
        # check that the type is int
        assert all(isinstance(n, int) for n in self.num_frames), "self.num_frames should be a list of int"

        self.n_files = len(self.file_names)

    def __eq__(self, other):
        if isinstance(other, FileManager):
            is_same = [
                self.data_dir == other.data_dir,
                self.file_names == other.file_names,
                self.loader == other.loader,
                self.num_frames == other.num_frames,
                self.n_files == other.n_files
            ]

            return np.all(is_same)
        else:
            print(f"__eq__ is Not Implemented for {FileManager} and {type(other)}")
            return NotImplemented

    def find_files(self, file_extension):
        """
        Searches for files , ending with file_extension in the data_dir
        :param file_extension: extension of files to search for
        :type file_extension: str
        :return: File names ( with extension, names only, no full path)
        :rtype: list[str]
        """
        files = list(self.data_dir.glob(f"*{file_extension}"))
        file_names = [file.name for file in files]
        return file_names

    # TODO : probably rename it?
    def check_files(self, file_names):
        """
        Given a list files checks that files are in the self.data_dir.
        :param file_names: list of filenames to check
        :type file_names: list[str]
        :return: the files ( full paths to the files ) and file_names
        :rtype: (Union(list[str],list[Path]),list[str])
        """
        files = [self.data_dir.joinpath(file) for file in file_names]
        for file in files:
            assert file.is_file(), f"File {file} is not found"
        return files, file_names

    def get_frames_per_file(self):
        """
        Get the number of frames  per file.
        returns a list with number fof frames per file.
        Expand this method if you want to work with other file types (not tiffs).
        """
        frames_per_file = []
        for file in self.file_names:
            n_frames = self.loader.get_frames_in_file(self.data_dir.joinpath(file))
            frames_per_file.append(n_frames)
        return frames_per_file

    # TODO : make sure it works
    # def change_files_order(self, order):
    #     """
    #     Changes the order of the files. If you notices that files are in the wrong order, provide the new order.
    #     If you wish to exclude any files, get rid of them ( don't include their IDs into the new order ).
    #
    #     :param order: The new order in which the files follow. Refer to file by it's position in the original list.
    #     Should be the same length as the number of files in the original list, or smaller (if you want to get rid of
    #     some files).
    #     :type order: list[int]
    #     """
    #     assert len(np.unique(order)) > self.n_files, \
    #         "Number of unique files is smaller than elements in the list! "
    #
    #     self.file_names = [self.file_names[i] for i in order]
    #     self.num_frames = [self.num_frames[i] for i in order]
    #
    # def __str__(self):
    #     description = f"Total of {self.n_files} files.\nCheck the order :\n"
    #     for i_file, file in enumerate(self.file_names):
    #         description = description + "[ " + str(i_file) + " ] " + file + " : " + str(
    #             self.num_frames[i_file]) + " frames\n"
    #     return description
    #
    # def __repr__(self):
    #     return self.__str__()


class TimeLabel:
    """
    Describes a particular time-located event during the experiment.
    Any specific aspect of the experiment that you want to document :
        temperature|light|sound|image on the screen|drug|behaviour ... etc.
    """

    def __init__(self, name, description=None, group=None):
        """

        :param name: the name for the time label. This is a unique identifier of the label.
                    Different labels must have different names.
                    Different labels are compared based on their names, so the same name means it is the same event.
        :type name: str

        :param description: a detailed description of the label. This is to give you more info, but it is not used for
        anything else.
        :type description: str

        :param group: the group that the label belongs to.
        :type group: str
        """
        self.name = name
        self.group = group
        self.description = description

    def __str__(self):
        """
        :return:
        """
        description = self.name
        if self.description is not None:
            description = description + " : " + self.description
        if self.group is not None:
            description = description + ". Group: " + self.group
        return description

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash((self.name, self.description))

    def __eq__(self, other):
        if isinstance(other, TimeLabel):
            # comparing by name
            same_name = self.name == other.name
            if self.group is not None or other.group is not None:
                same_group = self.group == other.group
                return same_name and same_group
            else:
                return same_name
        else:
            print(f"__eq__ is Not Implemented for {TimeLabel} and {type(other)}")
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_dict(self):
        d = {'name': self.name}
        if self.group is not None:
            d['group'] = self.group
        if self.description is not None:
            d['description'] = self.description
        return d

    @classmethod
    def from_dict(cls, d):
        description = None
        group = None
        if 'description' in d:
            description = d['description']
        if 'group' in d:
            group = d['group']
        return cls(d['name'], description=description, group=group)


class Labels:
    """
    Describes a particular group of time labels.
    TODO : also responsible for colors for plotting these labels.
    """

    def __init__(self, group, state_names, group_info=None, state_info={}):
        """
        group : str, the name of the group
        group_info : str, description of what this group is about
        states: list[str], the state names
        state_info: {state name : description}
        """

        self.group = group
        self.group_info = group_info
        self.state_names = state_names
        # create states
        self.states = []
        for state_name in self.state_names:
            if state_name in state_info:
                state = TimeLabel(state_name, description=state_info[state_name], group=self.group)
                setattr(self, state_name, state)
            else:
                state = TimeLabel(state_name, group=self.group)
                setattr(self, state_name, state)
            self.states.append(state)

    def __eq__(self, other):
        if isinstance(other, Labels):
            is_same = [
                self.group == other.group,
                self.group_info == other.group_info,
                self.state_names == other.state_names,
                self.states == other.states
            ]

            return np.all(is_same)
        else:
            print(f"__eq__ is Not Implemented for {Labels} and {type(other)}")
            return NotImplemented

    def __str__(self):
        description = f"Label group : {self.group}\n"
        description = description + f"States:\n"
        for state_name in self.state_names:
            description = description + f"{getattr(self, state_name)}\n"
        return description

    def __repr__(self):
        return self.__str__()


class Cycle:
    """
    Information about the repeated cycle of labels. Use it when you have some periodic conditions, like : light
    on , light off, light on, light off... will be made of list of labels [light_on, light_off] that repeat ..."""

    def __init__(self, label_order, timing):
        """
        :param label_order: a list of labels in the right order in which they follow
        :type label_order: list[TimeLabel]

        :param timing: timing of the corresponding labels, in frames (based on your imaging). Note that these are
        frames, not volumes !
        :type timing: list[int]
        """
        # check that all labels are from the same group
        label_group = label_order[0].group
        for label in label_order:
            assert label.group == label_group, \
                f"All labels should be from the same group, but got {label.group} and {label_group}"

        # check that timing is int
        assert all(isinstance(n, int) for n in timing), "timing should be a list of int"

        self.name = label_group
        self.label_order = label_order
        self.timing = list(timing)
        self.full_length = sum(self.timing)
        # list the length of the cycle, each element is the TimeLabel
        # TODO : turn it into an index ?
        self.per_frame_list = self.get_label_per_frame()

    def __eq__(self, other):
        if isinstance(other, Cycle):
            is_same = [
                self.name == other.name,
                self.label_order == other.label_order,
                self.timing == other.timing,
                self.full_length == other.full_length,
                self.per_frame_list == other.per_frame_list
            ]

            return np.all(is_same)
        else:
            print(f"__eq__ is Not Implemented for {Cycle} and {type(other)}")
            return NotImplemented

    def get_label_per_frame(self):
        """
        A list of labels per frame for one cycle only.
        :return: labels per frame for one full cycle
        :rtype: list[TimeLabels]
        """
        per_frame_label_list = []
        for (label_time, label) in zip(self.timing, self.label_order):
            per_frame_label_list.extend(label_time * [label])
        return per_frame_label_list

    def __str__(self):
        description = f"Cycle : {self.name}\n"
        description = description + f"Length: {self.full_length}\n"
        for (label_time, label) in zip(self.timing, self.label_order):
            description = description + f"Label {label.name}: for {label_time} frames\n"
        return description

    def __repr__(self):
        return self.__str__()

    def fit_frames(self, n_frames):
        """
        Calculates how many cycles you need to fully cover n_frames.
        :param n_frames: number of frames to cover
        :type n_frames: int
        :return: number of cycle
        :rtype: int
        """
        n_cycles = int(np.ceil(n_frames / self.full_length))
        return n_cycles

    def fit_labels_to_frames(self, n_frames):
        """
        Create a list of labels corresponding to each frame in the range of n_frames
        :param n_frames: number of frames to fit labels to
        :type n_frames: int
        :return: label_per_frame_list
        :rtype: list[TimeLabel]
        """
        n_cycles = self.fit_frames(n_frames)
        label_per_frame_list = np.tile(self.per_frame_list, n_cycles)
        # crop the tail
        return list(label_per_frame_list[0:n_frames])

    def fit_cycles_to_frames(self, n_frames):
        """
        Create a list of integers (what cycle iteration it is) corresponding to each frame in the range of n_frames
        :param n_frames: number of frames to fit cycle iterations to
        :type n_frames: int
        :return: cycle_per_frame_list
        :rtype: list[int]
        """
        n_cycles = self.fit_frames(n_frames)
        cycle_per_frame_list = []
        for n in np.arange(n_cycles):
            cycle_per_frame_list.extend([int(n)] * self.full_length)
        # crop the tail
        return cycle_per_frame_list[0:n_frames]

    def to_dict(self):
        label_order = [label.to_dict() for label in self.label_order]
        d = {'timing': self.timing, 'label_order': label_order}
        return d

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d):
        label_order = [TimeLabel.from_dict(ld) for ld in d['label_order']]
        return cls(label_order, d['timing'])

    @classmethod
    def from_json(cls, j):
        """
        j : json string
        """
        d = json.loads(j)
        return cls.from_dict(d)


class FrameManager:
    """
    Deals with frames. Which frames correspond to a volume / cycle/ condition.

    :param file_manager: info about the files.
    :type file_manager: FileManager
    """

    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.frame_to_file, self.frame_in_file = self.get_frame_mapping()

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
    def from_dir(cls, data_dir, file_names=None, frames_per_file=None):
        file_manager = FileManager(data_dir, file_names=file_names, frames_per_file=frames_per_file)
        return cls(file_manager)

    def get_frame_mapping(self):
        """
        Calculate frame range in each file.
        Returns a dict with file index for each frame and frame index in the file.
        Used to figure out in which stack the requested frames is.
        Frame number starts at 0, it is a numpy array or a list.

        :return: Dictionary mapping frames to files. 'file_idx' is a list of length equal to the total number of
        frames in all the files, where each element corresponds to a frame and contains the file index, where that
        frame is located. 'in_file_frame' is a list of length equal to the total number of
        frames in all the files, where each element corresponds to the index of the frame inside the file.

        :rtype: dict of str: list[int]
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
    Figures out how to get full volumes for certain time points.

    :param fpv: frames per volume, number of frames in one volume
    :type fpv: int

    :param fgf: first good frame, the first frame in the imaging session that is at the top of a volume.
    For example if you started imaging at the top of the volume, fgf = 0,
    but if you started somewhere in the middle, the first good frame is , for example, 23 ...
    :type fgf: int

    :param frame_manager: the info about the frames
    :type frame_manager: FrameManager
    """

    def __init__(self, fpv, frame_manager, fgf=0):
        # maybe I should do type checking automatically, something like here:
        # https://stackoverflow.com/questions/9305751/how-to-force-ensure-class-attributes-are-a-specific-type
        assert isinstance(fpv, int) or (isinstance(fpv, float) and fpv.is_integer()), "fpv must be an integer"
        assert isinstance(fgf, int) or (isinstance(fgf, float) and fgf.is_integer()), "fgf must be an integer"

        # frames per volume
        self.fpv = int(fpv)

        # get total number of frames
        self.frame_manager = frame_manager
        self.file_manager = frame_manager.file_manager
        self.n_frames = np.sum(self.file_manager.num_frames)

        # prepare info about frames at the beginning, full volumes and frames at the end
        # first good frame, start counting from 0 : 0, 1, 2, 3, ...
        # n_head is the number of frames before the first frame of the first full volume
        # n_tail is the number of frames after the last frame of the last full volume
        self.n_head = int(fgf)
        full_volumes, n_tail = divmod((self.n_frames - self.n_head), self.fpv)
        self.full_volumes = int(full_volumes)
        self.n_tail = int(n_tail)

        # map frames to slices an full volumes:
        self.frame_to_z = self.get_frames_to_z_mapping()
        self.frame_to_vol = self.get_frames_to_volumes_mapping()

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

    def get_frames_to_z_mapping(self):
        z_per_frame_list = np.arange(self.fpv).astype(int)
        # set at what z the imaging starts and ends
        i_from = self.fpv - self.n_head
        i_to = self.n_tail - self.fpv
        # map frames to z
        frame_to_z = np.tile(z_per_frame_list, self.full_volumes + 2)[i_from:i_to]
        return frame_to_z.tolist()

    def get_frames_to_volumes_mapping(self):
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
        return description

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_dir(cls, data_dir, fpv, fgf=0, file_names=None, frames_per_file=None):
        file_manager = FileManager(data_dir, file_names=file_names, frames_per_file=frames_per_file)
        frame_manager = FrameManager(file_manager)
        return cls(fpv, frame_manager, fgf=fgf)


class Annotation:
    """
    Time annotation of the experiment.
    """

    def __init__(self, n_frames, labels, frame_to_label, info=None,
                 cycle=None, frame_to_cycle=None):
        """
        Either frame_to_label_dict or n_frames need to be provided to infer the number of frames.
        If both are provided , they need to agree.

        :param labels: Labels
        :param info: str, description of the annotation
        :param frame_to_label: list[TimeLabels] what label it is for each frame
        :param frame_to_cycle: list[int] what cycle it is for each frame
        :param cycle: for annotation from cycles keeps the cycle
        :param n_frames: total number of frames, will be inferred from frame_to_label if not provided
        """
        # get total experiment length in frames, check that it is consistent
        if frame_to_label is not None:
            assert n_frames == len(frame_to_label), f"The number of frames in the frame_to_label," \
                                                    f"{len(frame_to_label)}," \
                                                    f"and the number of frames provided," \
                                                    f"{n_frames}, do not match."
        self.n_frames = n_frames
        self.frame_to_label = frame_to_label
        self.labels = labels
        self.name = self.labels.group
        self.info = info

        self.cycle = None
        # None if the annotation is not from a cycle
        if frame_to_cycle is not None:
            # check that frame_to_cycle is int
            assert all(isinstance(n, int) for n in frame_to_cycle), "frame_to_cycle should be a list of int"
            assert cycle is not None, "You have to provide cycle " \
                                      "if you are providing a frame_to_cycle for the Annotation"
        if cycle is not None:
            assert frame_to_cycle is not None, "You have to provide frame_to_cycle " \
                                               "if you are providing a cycle for the Annotation"
            assert n_frames == len(frame_to_cycle), f"The number of frames in the frame_to_cycle," \
                                                    f"{len(frame_to_cycle)}," \
                                                    f"and the number of frames provided," \
                                                    f"{n_frames}, do not match."
            self.cycle = cycle
            self.frame_to_cycle = frame_to_cycle

    def __eq__(self, other):
        if isinstance(other, Annotation):
            is_same = [
                self.n_frames == other.n_frames,
                self.frame_to_label == other.frame_to_label,
                self.labels == other.labels,
                self.name == other.name,
                self.info == other.info
            ]
            if self.cycle is not None:
                if other.cycle is None:
                    return False
                else:
                    is_same.extend([self.cycle == other.cycle,
                                    self.frame_to_cycle == other.frame_to_cycle])
            return np.all(is_same)
        else:
            print(f"__eq__ is Not Implemented for {Annotation} and {type(other)}")
            return NotImplemented

    @classmethod
    def from_cycle(cls, n_frames, labels, cycle, info=None):

        frame_to_label = cycle.fit_labels_to_frames(n_frames)
        frame_to_cycle = cycle.fit_cycles_to_frames(n_frames)
        return cls(n_frames, labels, frame_to_label, info=info,
                   cycle=cycle, frame_to_cycle=frame_to_cycle)

    @classmethod
    def from_timing(cls, n_frames, labels, label_order, timing, info=None):
        assert n_frames == sum(timing), "number of frames and total timing should be the same"
        # make a fake cycle the length of the whole recording
        frame_to_label = Cycle(label_order, timing).per_frame_list
        return cls(n_frames, labels, frame_to_label, info=info)

    def __str__(self):
        description = f"Annotation type: {self.name}\n"
        if self.info is not None:
            description = description + f"{self.info}\n"
        description = description + f"Total frames : {self.n_frames}\n"
        return description

    def __repr__(self):
        return self.__str__()


class Experiment:
    """
    Information about the experiment.
    Will use all the information you provided to figure out what frames to give you based on your request.
    """

    def __init__(self, db_reader):
        """
        :param db_reader: DbReader connected to the data base with the experiment description
        """
        self.db = db_reader
        # will add the loader the first time you are loading anything
        # in load_frames() or load_volumes()
        self.loader = None

    @classmethod
    def create(cls, volume_manager, annotations):
        """
        Creates a database instance and initialises the experiment.
        provide cycles if you have any cycles
        """
        db = DbWriter.create()
        db.populate(volumes=volume_manager, annotations=annotations)
        db_reader = DbReader(db.connection)
        return cls(db_reader)

    def save(self, file_name):
        """
        Saves a database into a file.
        :param file_name: full path to a file to save database.
        (Usually the filename would end with .db)
        :type file_name: Union(Path, str)
        """
        self.db.save(file_name)

    @classmethod
    def load(cls, file_name):
        """
        Saves a database into a file.
        :param file_name: full path to a file to load database.
        :type file_name: Union(Path, str)
        """
        db_reader = DbReader.load(file_name)
        return cls(db_reader)

    def choose_frames(self, conditions, between_group_logic="and"):
        """
        Selects the frames that correspond to specified conditions;
        "or" inside the label-group and "or" or "and" between the labels groups,
         defined by between_group_logic.
        To load the selected frames, use load_frames()

        :param conditions: a list of conditions on the annotation labels
        in a form [(group, name),(group, name), ...] where group is a string for the annotation type
        and name is the name of the label of that annotation type. For example [('light', 'on'), ('shape','c')]
        :type conditions: [tuple]
        :param between_group_logic: "and" or "or" , default is "and".
        :type between_group_logic: str
        :return: list of frame ids that were chosen.
                Remember that frame numbers start at 1.
        """
        assert between_group_logic == "and" or between_group_logic == "or", \
            'between_group_logic should be equal to "and" or "or"'
        frames = []
        if between_group_logic == "and":
            frames = self.db.get_and_frames_per_annotations(conditions)
        elif between_group_logic == "or":
            frames = self.db.get_or_frames_per_annotations(conditions)

        return frames

    def choose_volumes(self, conditions, between_group_logic="and"):
        """
        Selects only full volumes that correspond to specified conditions;
        "or" inside the label-group and "or" or "and" between the labels groups,
         defined by between_group_logic.
        To load the selected volumes, use load_volumes()

        :param conditions: a list of conditions on the annotation labels
        in a form [(group, name),(group, name), ...] where group is a string for the annotation type
        and name is the name of the label of that annotation type. For example [('light', 'on'), ('shape','c')]
        :type conditions: [tuple]
        :param between_group_logic: "and" or "or" , default is "and".
        :type between_group_logic: str
        :return: list of volumes and list of frame ids that were chosen.
                Remember that frame numbers start at 1, but volumes start at 0.
                TODO : make everything start at 1 ????
        """
        # get all the frames that correspond to the conditions
        frames = self.choose_frames(conditions, between_group_logic=between_group_logic)
        n_frames = len(frames)
        # leave only such frames that correspond to full volumes
        # TODO : why do I even need to return frames?
        volumes, frames = self.db.choose_full_volumes(frames)
        n_dropped = n_frames - len(frames)
        print(f"Choosing only full volumes. "
              f"Dropped {n_dropped} frames, kept {len(frames)}")

        return volumes

    def load_volumes(self, volumes):
        """
        Load volumes.
        """
        frames = self.db.get_frames_per_volumes(volumes)
        info = self.db.prepare_frames_for_loading(frames)
        # unpack
        data_dir, file_names, file_ids, frame_in_file, volumes = info
        # make full paths to files ( remember file ids start with 1 )
        files = [Path(data_dir, file_names[file_id - 1]) for file_id in file_ids]
        if self.loader is None:
            self.loader = ImageLoader(Path(data_dir, file_names[0]))
        volumes_img = self.loader.load_volumes(frame_in_file,
                                               files,
                                               volumes,
                                               show_file_names=False,
                                               show_progress=True)
        return volumes_img

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    def summary(self):
        """
        Prints a detailed description of the experiment.
        """
        raise NotImplementedError

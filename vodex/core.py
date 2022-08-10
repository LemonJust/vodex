"""
Classes to specify the experimental conditions and load necessary data.
"""
from tifffile import TiffFile
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sqlite3 import connect


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
            # comparing by name
            same_supported_extension = self.supported_extension == other.supported_extension
            same_file_extension = self.file_extension == other.file_extension
            same_loader = self.loader == other.loader

            return np.all([same_supported_extension,
                           same_file_extension,
                           same_loader])
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

    def load_volumes(self, frames, files, volumes, show_file_names=False, show_progress=True):
        """
        Loads specified frames from specified files and shapes them into volumes.
        :param frames: list of frames IN FILES to load.
        :type frames: list[int]
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
        frames = self.loader.load_frames(frames, files,
                                         show_file_names=show_file_names,
                                         show_progress=show_progress)
        n_frames, w, h = frames.shape

        # get volume information
        i_volume, count = np.unique(volumes, return_counts=True)
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
            # comparing by name
            same_data_dir = self.data_dir == other.data_dir
            same_file_names = self.file_names == other.file_names
            same_loader = self.loader == other.loader
            same_num_frames = self.num_frames == other.num_frames
            same_n_files = self.n_files == other.n_files

            return np.all([same_data_dir,
                           same_file_names,
                           same_loader,
                           same_num_frames,
                           same_n_files])
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
        """
        Compares the timelabel to another timelabel.
        The names need to match ( compared as strings ).
        When the timelabel has a group , both timelabels need to have a group
        and both names and group need to match.

        :param other: TimeLabel or string to compare to
        :return: True or False , whether the two Timelabels are the same
        """
        # comparing to other TimeLabel
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
            # comparing by name
            same_name = self.name == other.name
            same_label_order = self.label_order == other.label_order
            same_timing = self.timing == other.timing
            same_full_length = self.full_length == other.full_length
            same_per_frame_list = self.per_frame_list == other.per_frame_list

            return np.all([same_name,
                           same_label_order,
                           same_timing,
                           same_full_length,
                           same_per_frame_list])
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
            # comparing by name
            same_file_manager = self.file_manager == other.file_manager
            same_frame_to_file = self.frame_to_file == other.frame_to_file
            same_frame_in_file = self.frame_in_file == other.frame_in_file

            return np.all([same_file_manager,
                           same_frame_to_file,
                           same_frame_in_file])
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
        for vol in np.arange(self.full_volumes).astype(int):
            frame_to_vol.extend([vol] * self.fpv)
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
        # TODO : make it better, this looks scary
        if isinstance(other, Annotation):
            # comparing by name
            same_n_frames = self.n_frames == other.n_frames
            same_frame_to_label = self.frame_to_label == other.frame_to_label
            same_labels = self.labels == other.labels
            same_name = self.name == other.name
            same_info = self.info == other.info
            if self.cycle is not None:
                if other.cycle is None:
                    return False
                else:
                    same_cycle = self.cycle == other.cycle
                    same_frame_to_cycle = self.frame_to_cycle == other.frame_to_cycle
                    return np.all([same_n_frames,
                                   same_frame_to_label,
                                   same_labels,
                                   same_name,
                                   same_cycle,
                                   same_frame_to_cycle])
            return np.all([same_n_frames,
                           same_frame_to_label,
                           same_labels,
                           same_name])
        else:
            print(f"__eq__ is Not Implemented for {Annotation} and {type(other)}")
            return NotImplemented

    @classmethod
    def from_cycle(cls, n_frames, labels, cycle, info=None):

        frame_to_label = cycle.fit_labels_to_frames(n_frames)
        frame_to_cycle = cycle.fit_cycles_to_frames(n_frames)
        return cls(n_frames, labels, frame_to_label, info=info,
                   cycle=cycle, frame_to_cycle=frame_to_cycle)

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

    def __init__(self, db):
        """
        :param volume_manager: volume manager
        :param annotation: annotation
        :param cycles: cycles ( optional - if you want to keep cycles information )
        """
        self.db = db

    @classmethod
    def create(cls, volume_manager, annotation, cycles=None):
        """
        Creates a database instance and initialises the experiment.
        provide cycles if you have any cycles
        """

        frame_manager = volume_manager.frame_manager
        file_manager = frame_manager.file_manager

        assert frame_manager.n_frames == annotation.n_frames, \
            f"Number of frames in the frame manager, {frame_manager.n_frames}, " \
            f"doesn't match the number of frames in the annotation , {annotation.n_frames}"

        db = DbManager.initialise_db()
        db.create_tables()

        db.populate_Options(file_manager, volume_manager)

    def choose_frames(self, labels_dict, between_group_logic="and"):
        """
        Selects the frames that correspond to specified conditions;
        "or" inside the label-group and "or" or "and" between the labels groups, defined by between_group_logic.
        To actually load the selected frames, use frame_manager.load_frames()

        :param labels_dict: dict {label group : label list}
        :param between_group_logic: "and" or "or" , default is "and"
        :return: frame_ids, list of frame ids that were chosen
        """
        assert between_group_logic == "and" or between_group_logic == "or", \
            'between_group_logic should be equal to "and" or "or"'

        def choose_inside_group(labels_dict, label_group, annotation):
            is_group_label = False
            for label in labels_dict[label_group]:
                # logical or inside the group
                is_group_label = np.logical_or(is_group_label,
                                               annotation.frame_to_label[label_group] == label)
            return is_group_label

        if between_group_logic == "or":
            chosen_frames = False
            # choose frames with "or" logic inside group
            for label_group in labels_dict:
                is_group_label = choose_inside_group(labels_dict, label_group, self.annotation)
                # logical "or" between the groups
                chosen_frames = np.logical_or(chosen_frames, is_group_label)

        if between_group_logic == "and":
            chosen_frames = True
            # choose frames with "or" logic inside group
            for label_group in labels_dict:
                is_group_label = choose_inside_group(labels_dict, label_group, self.annotation)
                # logical "and" between the groups
                chosen_frames = np.logical_and(chosen_frames, is_group_label)

        frame_ids = np.where(chosen_frames)[0]
        return frame_ids

    def choose_volumes(self, labels_dict, between_group_logic="and",
                       return_partial_volumes=False, return_counts=False):
        """
        Selects the volumes that correspond to specified conditions;
        "or" inside the label-group and "or" or "and" between the labels groups, defined by between_group_logic.
        To actually load the selected volumes, use volume_manager.load_volumes()

        :param labels_dict: dict {label group : label list}
        :param between_group_logic: "and" or "or" , default is "and"
        :param return_counts: bool, wether or not to return counts. If false, returns only full volumes (if return_counts is False ) or
        all the volumes and counts (if return_counts is True ). If True , ignores return_partial_volumes.
        :param return_partial_volumes: bool, wether or not to return partial volumes.
        If True, returns full and partial volumes separately.
        If false, returns depend on return_counts.
        :return:
        """
        # TODO : make returns clearer !
        frame_ids = self.choose_frames(labels_dict, between_group_logic=between_group_logic)

        vol_ids = np.array(self.volume_manager.frame_to_vol)[frame_ids]
        volumes, counts = np.unique(vol_ids, return_counts=True)

        # get frames that correspond to full volumes and partial volumes
        full_volumes = volumes[counts == self.volume_manager.fpv]
        partial_volumes = volumes[counts < self.volume_manager.fpv]

        # figure out returns
        if return_counts:
            return volumes, counts
        if return_partial_volumes:
            return full_volumes, partial_volumes
        return full_volumes

    def choose_slices(self):
        """
        Selects the slices that correspond to specified conditions;
        """
        raise NotImplementedError

    # def select_zslices(self, zslice, condition=None):
    #     """
    #     Selects the frames for a specific zslice that correspond to a specified condition.
    #     If condition is None, selects all the frames for the specified zslice.
    #     To actually load the selected frames, use frame_manager.load_frames()
    #
    #     :param zslice: the zslice for which to select the frames
    #     :type zslice: int
    #
    #     :param condition: the condition for which to select the frames, or None
    #     :type condition: Condition
    #
    #     :return: 1D array of indices: a list of frames corresponding to the zslice and the condition.
    #     :rtype: numpy.ndarray
    #     """
    #     zslice_match = self.volume_manager.frame_to_z == zslice
    #
    #     if condition is not None:
    #         condition_match = self.frame_to_condition == condition
    #         request_met = np.logical_and(condition_match, zslice_match)
    #     else:
    #         request_met = zslice_match
    #     which_frames = np.where(request_met)[0]
    #     return which_frames

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    def summary(self):
        """
        Prints a detailed description of the experiment.
        """
        raise NotImplementedError


class DbManager:
    """
    Database interface that abstracts the SQLite calls
    Code adopted from https://www.devdungeon.com/content/python-sqlite3-tutorial
    Maybe better stuff here : https://www.sqlitetutorial.net/sqlite-python/creating-tables/

    Some thoughts on try-catch :
    https://softwareengineering.stackexchange.com/questions/64180/good-use-of-try-catch-blocks
    """

    def __init__(self, db):
        self.connection = db
        db.execute("PRAGMA foreign_keys = 1")

    def list_tables(self):
        """
        Shows all the tables in the database
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        print(cursor.fetchall())
        cursor.close()

    def table_as_df(self, table_name):
        """
        Returns the whole table as a dataframe
        table_name : name of the table you want to see
        """

        cursor = self.connection.cursor()
        try:
            query = cursor.execute(f"SELECT * From {table_name}")
            cols = [column[0] for column in query.description]
            df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        except Exception as e:
            print(f"Could not show {table_name} because {e}")
            raise e
        finally:
            cursor.close()
        return df

    def get_n_frames(self):
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM Frames")
            n_frames = cursor.fetchone()[0]
        except Exception as e:
            print(f"Could not get total number of frames from Frames because {e}")
            raise e
        finally:
            cursor.close()
        return n_frames

    def save(self, file_name):
        """
        Backup a database to a file.
        Will CLOSE connection to the database in memory!
        file_name : "databasename.db"
        """

        def progress(status, remaining, total):
            print(f'Copied {total - remaining} of {total} pages...')

        backup_db = connect(file_name)
        with backup_db:
            self.connection.backup(backup_db, progress=progress)
        self.connection.close()
        backup_db.close()

    @classmethod
    def empty(cls):
        """
        Creates an empty DB to the experiment in memory
        """
        # For an in-memory only database:
        memory_db = connect(':memory:')
        return cls(memory_db)

    @classmethod
    def load(cls, file_name):
        """
        Load the contents of a database file on disk to a
        transient copy in memory without modifying the file
        """
        disk_db = connect(file_name)
        memory_db = connect(':memory:')
        disk_db.backup(memory_db)
        disk_db.close()
        # Now use `memory_db` without modifying disk db
        return cls(memory_db)

    def populate(self, files=None, frames=None, volumes=None, annotations=None):
        """
        Creates the tables if they don't exist and fills the provided data.

        :param files: information about the files : number of files, their location
        :type files: FileManager
        :param frames: mapping of frames to files
        :type frames: FrameManager
        :param volumes: mapping of frames to volumes, and to slices in volumes, frames per volume
        :type volumes: VolumeManager
        :param annotations: mapping of frames to labels, list of annotations
        :type annotations: [Annotation]
        :return: None
        """
        # TODO : add warnings when you are trying to write the same data again

        # will only create if they don't exist
        self._create_tables()
        # TODO : write cases for files and frames not None

        if volumes is not None:
            self._populate_Options(volumes.file_manager, volumes)
            self._populate_Files(volumes.file_manager)
            self._populate_Frames(volumes.frame_manager)
            self._populate_Volumes(volumes)

        if annotations is not None:
            for annotation in annotations:
                self._populate_AnnotationTypes(annotation)
                self._populate_AnnotationTypeLabels(annotation)
                self._populate_Annotations(annotation)
                if annotation.cycle is not None:
                    self._populate_Cycles(annotation)
                    self._populate_CycleIterations(annotation)

    def _create_tables(self):

        # TODO : change UNIQUE(a, b)
        # into primary key over both columns a and b where appropriate

        db_cursor = self.connection.cursor()

        sql_create_Options_table = """
            CREATE TABLE IF NOT EXISTS "Options" (
            "Key"	TEXT NOT NULL UNIQUE,
            "Value"	TEXT NOT NULL,
            "Description"	TEXT,
            PRIMARY KEY("Key")
            )
            """
        db_cursor.execute(sql_create_Options_table)

        sql_create_Files_table = """
            CREATE TABLE IF NOT EXISTS "Files" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "FileName"	TEXT NOT NULL UNIQUE,
            "NumFrames"	INTEGER NOT NULL,
            PRIMARY KEY("Id" AUTOINCREMENT)
            )
            """
        db_cursor.execute(sql_create_Files_table)

        sql_create_AnnotationTypes_table = """
            CREATE TABLE IF NOT EXISTS "AnnotationTypes" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "Name"	TEXT NOT NULL UNIQUE,
            "Description"	TEXT,
            PRIMARY KEY("Id" AUTOINCREMENT)
            )
            """
        db_cursor.execute(sql_create_AnnotationTypes_table)

        sql_create_Frames_table = """
            CREATE TABLE IF NOT EXISTS "Frames" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "FrameInFile"	INTEGER NOT NULL,
            "FileId"	INTEGER NOT NULL,
            PRIMARY KEY("Id" AUTOINCREMENT),
            FOREIGN KEY("FileId") REFERENCES "Files"("Id")
            )
            """
        db_cursor.execute(sql_create_Frames_table)

        sql_create_Cycles_table = """
            CREATE TABLE IF NOT EXISTS "Cycles" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "AnnotationTypeId"	INTEGER NOT NULL UNIQUE,
            "Structure"	TEXT NOT NULL,
            FOREIGN KEY("AnnotationTypeId") REFERENCES "AnnotationTypes"("Id"),
            PRIMARY KEY("Id" AUTOINCREMENT)
            )
            """
        db_cursor.execute(sql_create_Cycles_table)

        sql_create_AnnotationTypeLabels_table = """
            CREATE TABLE IF NOT EXISTS "AnnotationTypeLabels" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "AnnotationTypeId"	INTEGER NOT NULL,
            "Name"	TEXT NOT NULL,
            "Description"	TEXT,
            PRIMARY KEY("Id" AUTOINCREMENT),
            FOREIGN KEY("AnnotationTypeId") REFERENCES "AnnotationTypes"("Id"),
            UNIQUE("AnnotationTypeId","Name")
            )
            """
        db_cursor.execute(sql_create_AnnotationTypeLabels_table)

        sql_create_Annotations_table = """
            CREATE TABLE IF NOT EXISTS "Annotations" (
            "FrameId"	INTEGER NOT NULL,
            "AnnotationTypeLabelId"	INTEGER NOT NULL,
            FOREIGN KEY("FrameId") REFERENCES "Frames"("Id"),
            FOREIGN KEY("AnnotationTypeLabelId") REFERENCES "AnnotationTypeLabels"("Id"),
            UNIQUE("FrameId","AnnotationTypeLabelId")
            )
            """
        db_cursor.execute(sql_create_Annotations_table)

        sql_create_CycleIterations_table = """
            CREATE TABLE IF NOT EXISTS "CycleIterations" (
            "FrameId"	INTEGER NOT NULL,
            "CycleId"	INTEGER NOT NULL,
            "CycleIteration"	INTEGER NOT NULL,
            FOREIGN KEY("CycleId") REFERENCES "Cycles"("Id"),
            FOREIGN KEY("FrameId") REFERENCES "Frames"("Id")
            )
            """
        db_cursor.execute(sql_create_CycleIterations_table)

        sql_create_Volumes_table = """
            CREATE TABLE IF NOT EXISTS "Volumes" (
            "FrameId"	INTEGER NOT NULL UNIQUE,
            "VolumeId"	INTEGER NOT NULL,
            "SliceInVolume"	INTEGER NOT NULL,
            PRIMARY KEY("FrameId" AUTOINCREMENT),
            FOREIGN KEY("FrameId") REFERENCES "Frames"("Id")
            UNIQUE("VolumeId","SliceInVolume")
            )
            """
        db_cursor.execute(sql_create_Volumes_table)

        db_cursor.close()

    def _populate_Options(self, file_manager, volume_manager):
        """
        Options: dictionary with key - value pairs
        another way of dealing with Errors : ( more pretty ??? )
        https://www.w3resource.com/python-exercises/sqlite/python-sqlite-exercise-6.php
        """
        row_data = [("data_dir", file_manager.data_dir.as_posix()),
                    ("frames_per_volume", volume_manager.fpv),
                    ("first_good_frame", volume_manager.fpv),
                    ("num_tail_frames", volume_manager.n_tail),
                    ("num_full_volumes", volume_manager.full_volumes)]

        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO Options (Key, Value) VALUES (?, ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to Options because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_Files(self, file_manager):
        """
        file_name : list with filenames per file (str)
        num_frames : list with number of frames per file (int)
        """
        row_data = [(filename, frames) for
                    filename, frames in zip(file_manager.file_names, file_manager.num_frames)]

        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO Files (FileName, NumFrames) VALUES (?, ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to Files because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_Frames(self, frame_manager):
        """
        Something like :
        insert into tab2 (id_customers, value)
        values ((select id from tab1 where customers='john'), 'alfa');
        but in SQLite
        https://www.tutorialspoint.com/sqlite/sqlite_insert_query.htm
        """
        # adding +1 since the frame_to_file is indexing files from 0 and sqlite gave files IDs from 1
        row_data = [(frame_in_file, frame_to_file + 1) for
                    frame_in_file, frame_to_file in zip(frame_manager.frame_in_file, frame_manager.frame_to_file)]

        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO Frames (FrameInFile, FileId) VALUES (?, ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to Frames because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_Volumes(self, volume_manager):

        row_data = [(volume_id, slice_in_volume) for
                    volume_id, slice_in_volume in zip(volume_manager.frame_to_vol, volume_manager.frame_to_z)]

        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO Volumes (VolumeId, SliceInVolume) VALUES (?, ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to Volumes because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_AnnotationTypes(self, annotation):
        """
        """
        row_data = (annotation.name, annotation.info)
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "INSERT INTO AnnotationTypes (Name, Description) VALUES (?, ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to AnnotationTypes because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_AnnotationTypeLabels(self, annotation):

        row_data = [(label.group, label.name, label.description)
                    for label in annotation.labels.states]

        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO AnnotationTypeLabels (AnnotationTypeId, Name, Description) " +
                "VALUES((SELECT Id FROM AnnotationTypes WHERE Name = ?), ?, ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to AnnotationTypeLabels because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_Annotations(self, annotation):
        n_frames = self.get_n_frames()
        assert n_frames == annotation.n_frames, f"Number of frames in the annotation, {annotation.n_frames}," \
                                                f"doesn't match" \
                                                f" the expected number of frames {n_frames}"
        frames = range(n_frames)
        row_data = [(frame + 1, label.name, label.group)
                    for frame, label in zip(frames, annotation.frame_to_label)]
        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO Annotations (FrameId, AnnotationTypeLabelId) " +
                "VALUES(?, (SELECT Id FROM AnnotationTypeLabels "
                "WHERE Name = ? " +
                "AND AnnotationTypeId = (SELECT Id FROM AnnotationTypes " +
                "WHERE Name = ?)))",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write group {annotation.name} to Annotations because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_Cycles(self, annotation):
        """
        """
        row_data = (annotation.name, annotation.cycle.to_json())
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "INSERT INTO Cycles (AnnotationTypeId, Structure) " +
                "VALUES((SELECT Id FROM AnnotationTypes WHERE Name = ?), ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to Cycles because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_CycleIterations(self, annotation):
        n_frames = self.get_n_frames()
        assert n_frames == annotation.n_frames, f"Number of frames in the annotation, {annotation.n_frames}," \
                                                f"doesn't match" \
                                                f" the expected number of frames {n_frames}"
        # get cycle id by annotation type
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT Id FROM Cycles " +
            "WHERE AnnotationTypeId = (SELECT Id FROM AnnotationTypes " +
            "WHERE Name = ?)", (annotation.name,))
        cycle_id = cursor.fetchone()[0]
        cursor.close()

        # prepare rows
        frames = range(n_frames)
        row_data = [(frame + 1, cycle_id, iteration)
                    for frame, iteration in zip(frames, annotation.frame_to_cycle)]

        # insert into CycleIterations
        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO CycleIterations (FrameId, CycleId, CycleIteration) " +
                "VALUES(?, ?,?)", row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write group {annotation.name} to CycleIterations because {e}")
            raise e
        finally:
            cursor.close()


def test_creating_the_db():
    """
    Tests if the frames returned correspond to the conditions.
    """
    data_dir = r"D:\Code\repos\vodex\data\test\test_movie"
    fpv = 10

    # labels
    shape = Labels("shape", ["c", "s"],
                   state_info={"c": "circle on the screen", "s": "square on the screen"})
    light = Labels("light", ["on", "off"], group_info="Information about the light",
                   state_info={"on": "the intensity of the background is high",
                               "off": "the intensity of the background is low"})
    c_num = Labels("c label", ['c1', 'c2', 'c3'], state_info={'c1': 'written c1', 'c2': 'written c1'})
    n_frames = 1000

    volume_m = VolumeManager.from_dir(data_dir, fpv)
    # annotation
    shape_cycle = Cycle([shape.c, shape.s], [15, 15])
    light_cycle = Cycle([light.off, light.on], [15, 20])
    c_cycle = Cycle([c_num.c1, c_num.c2,
                     c_num.c3, c_num.c2], [10, 10, 10, 10])

    shape_annotation = Annotation.from_cycle(n_frames, shape, shape_cycle)
    light_annotation = Annotation.from_cycle(n_frames, er.light, light_cycle,
                                             info="a cycle off-on")
    c_annotation = Annotation.from_cycle(n_frames, c_num, c_cycle)

    db = DbManager.empty()
    db.populate(volumes=volume_m, annotations=[shape_annotation, light_annotation, c_annotation])
    db.save(r'D:\Code\repos\vodex\data\test\test_experiment2.db')


if __name__ == '__main__':
    # BadTester.test_file_manager()
    # BadTester.test_frame_manager()
    # BadTester.test_volume_manager()
    # BadTester.test_cycle()
    # BadTester.test_annotation()

    # BadTester.test_db_manager_population()
    # BadTester.test_db_manager_load()

    test_creating_the_db()

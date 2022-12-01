"""
This module contains classes to specify the information about the imaging data, about the experimental conditions
and load the data based on the specified information.

The module contains the following classes:

- `FileManager` - Figures out stuff concerning the many files. For example in what order do stacks go?
    Will grab all the files with the provided file_extension
    in the provided folder and order them alphabetically.

- `TimeLabel` - Describes a particular time-located event during the experiment.
        Any specific aspect of the experiment that you want to document :
        temperature|light|sound|image on the screen|drug|behaviour ... etc.

- `Labels` - Describes a particular group of time labels.

- `Cycle` - Information about the repeated cycle of labels. Use it when you have some periodic conditions, like : light
        on , light off, light on, light off... will be made of list of labels [light_on, light_off] that repeat ...

- `Timeline` - Information about the sequence of labels. Use it when you have non-periodic conditions.

- `FrameManager` - Deals with frames. Which frames correspond to a volume / cycle/ condition.

- `VolumeManager` - Figures out how to get full volumes for certain time points.

- `Annotation` - Time annotation of the experiment.

- `Experiment` - Information about the experiment.
        Will use all the information you provided to figure out what frames to give you based on your request.
"""
from itertools import groupby
import json
import numpy as np
from pathlib import Path
from tifffile import TiffFile
from tqdm import tqdm
from typing import Union
import warnings

from .loaders import ImageLoader
from .dbmethods import DbWriter, DbReader
from .utils import list_of_int


class FileManager:
    """
    Collects and stores the information about all the image files.
    By default, it will search for all the files with the provided file extension
    in the provided data director and order them alphabetically.
    If a list of file names is provided, will use these files in the provided order.

    Args:
        data_dir: path to the folder with the files, ends with "/" or "\\"
        file_names: names of files relative to the data_dir (TODO:?)
        frames_per_file: number of frames in each file.
        file_extension: file extension to search for (if files are not provided)

    Attributes:
        data_dir: the directory with all the imaging data
        file_names: names of files relative to the data_dir (TODO:?)
        loader: initialised image loader (see loaders.ImageLoader for more details)
        num_frames: a number of frames per file
        n_files: total number of image files
    """

    def __init__(self, data_dir: Union[str, Path], file_extension: str = ".tif",
                 file_names: list[str] = None, frames_per_file: list[int] = None):

        # 1. get data_dir and check it exists
        self.data_dir: Path = Path(data_dir)
        assert self.data_dir.is_dir(), f"No directory {self.data_dir}"

        # 2. get files
        if file_names is None:
            # if files are not provided , search for tiffs in the data_dir
            self.file_names: list[str] = self.find_files(file_extension)
        else:
            # if a list of files is provided, check it's in the folder
            self.file_names: list[str] = self.check_files(file_names)[1]
        # 3. Initialise ImageLoader
        #    will pick the image loader that works with the provided file type
        self.loader: ImageLoader = ImageLoader(self.data_dir.joinpath(self.file_names[0]))

        # 4. Get number of frames per file
        if frames_per_file is None:
            # if number of frames not provided , search for tiffs in the data_dir
            self.num_frames: list[int] = self.get_frames_per_file()
        else:
            # if provided ... we'll trust you - hope these numbers are correct
            self.num_frames: list[int] = frames_per_file
        # check that the type is int
        assert all(isinstance(n, (int, np.integer)) for n in self.num_frames), \
            "self.num_frames should be a list of int"

        self.n_files: int = len(self.file_names)

    def __str__(self):
        description = f"Image files information :\n\n"
        description = description + f"files directory: {self.data_dir}\n"
        description = description + f"files [number of frames]: \n"
        for (i_file, (file_name, num_frames)) in enumerate(zip(self.file_names, self.num_frames)):
            description = description + f"{i_file}) {file_name} [{num_frames}]\n"
        return description

    def __repr__(self):
        return self.__str__

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

    def find_files(self, file_extension: str) -> list[str]:
        """
        Searches for files ending with the provided file extension in the data directory.
        Args:
            file_extension: extension of files to search for
        Returns:
            A list of file names. File names are with the extension, relative to the data directory
            (names only, not full paths to files)
        """
        files = list(self.data_dir.glob(f"*{file_extension}"))
        file_names = [file.name for file in files]
        return file_names

    # TODO : probably rename it?
    def check_files(self, file_names: list[str]) -> (Union[list[str], list[Path]], list[str]):
        """
        Given a list of files checks that files are in the data directory.
        Args:
            file_names: list of filenames to check.
        Returns:
            If the files are all present in the directory,
            returns the files ( full paths to the files ) and file_names.
        """
        files = [self.data_dir.joinpath(file) for file in file_names]
        for file in files:
            assert file.is_file(), f"File {file} is not found"
        return files, file_names

    def get_frames_per_file(self) -> list[int]:
        """
        Get the number of frames per file.
        Returns:
            a list with number of frames per file.
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
    Any specific aspect of the experiment that you may want to document :
    temperature|light|sound|image on the screen|drug|behaviour ... etc.

    Args:
        name: the name for the time label. This is a unique identifier of the label.
                    Different labels must have different names.
                    Different labels are compared based on their names, so the same name means it is the same event.
        description: a detailed description of the label. This is to give you more info, but it is not used for
            anything else.
        group: the group that the label belongs to.

    Attributes:
        name: the name for the time label. This is a unique identifier of the label.
                    Different labels must have different names.
                    Different labels are compared based on their names, so the same name means it is the same event.
        description: a detailed description of the label. This is to give you more info, but it is not used for
            anything else.
        group: the group that the label belongs to.
    """

    def __init__(self, name: str, description: str = None, group: str = None):
        self.name: str = name
        self.group: str = group
        self.description: str = description

    def __str__(self):
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

    def to_dict(self) -> dict:
        """
        Put all the information about a TimeLabel object into a dictionary.
        Returns:
            a dictionary with fields 'name', 'group', 'description' storing the corresponding attributes.
        """
        d = {'name': self.name}
        if self.group is not None:
            d['group'] = self.group
        if self.description is not None:
            d['description'] = self.description
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Create a TimeLabel object from a dictionary.

        Returns:
            (TimeLabel): a TimeLabel object with attributes 'name', 'group', 'description'
            filled from the dictionary fields.
        """
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
    TODO : also responsible for colors for plotting these labels?

    Args:
        group : the name of the group
        group_info : description of what this group is about. Just for storing the information.
        state_names: the state names
        state_info: description of each individual state {state name : description}. Just for storing the information.
    Attributes:
        group (str): the name of the group
        group_info (str): description of what this group is about. Just for storing the information.
        state_names (list[str]): the state names
        states (list[TimeLabel]): list of states, each state as a TimeLabel object

    """

    def __init__(self, group: str, state_names: list[str], group_info: str = None, state_info: dict = None):

        if state_info is None:
            state_info = {}
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
    on , light off, light on, light off... will be made of list of labels [light_on, light_off]
    that repeat to cover the whole tie of the experiment. All labels must be from the same label group!
    Args:
        label_order: a list of labels in the right order in which they follow
        duration: duration of the corresponding labels, in frames (based on your imaging).
            Note that these are frames, not volumes !
    """

    def __init__(self, label_order: list[TimeLabel], duration: Union[np.array, list[int]]):
        # check that all labels are from the same group
        label_group = label_order[0].group
        for label in label_order:
            assert label.group == label_group, \
                f"All labels should be from the same group, but got {label.group} and {label_group}"

        # check that timing is int
        assert all(isinstance(n, (int, np.integer)) for n in duration), "timing should be a list of int"

        self.name = label_group
        self.label_order = label_order
        self.duration = list_of_int(duration)
        self.full_length = sum(self.duration)
        # list the length of the cycle, each element is the TimeLabel
        # TODO : turn it into an index ?
        self.per_frame_list = self.get_label_per_frame()

    def __eq__(self, other):
        if isinstance(other, Cycle):
            is_same = [
                self.name == other.name,
                self.label_order == other.label_order,
                self.duration == other.duration,
                self.full_length == other.full_length,
                self.per_frame_list == other.per_frame_list
            ]

            return np.all(is_same)
        else:
            print(f"__eq__ is Not Implemented for {Cycle} and {type(other)}")
            return NotImplemented

    def get_label_per_frame(self) -> list[TimeLabel]:
        """
        A list of labels per frame for one cycle only.
        Returns:
            labels per frame for one full cycle
        """
        per_frame_label_list = []
        for (label_time, label) in zip(self.duration, self.label_order):
            per_frame_label_list.extend(label_time * [label])
        return per_frame_label_list

    def __str__(self):
        description = f"Cycle : {self.name}\n"
        description = description + f"Length: {self.full_length}\n"
        for (label_time, label) in zip(self.duration, self.label_order):
            description = description + f"Label {label.name}: for {label_time} frames\n"
        return description

    def __repr__(self):
        return self.__str__()

    def fit_frames(self, n_frames: int) -> int:
        """
        Calculates how many cycles you need to fully cover n_frames.
        Args:
            n_frames: number of frames to cover
        Returns:
            number of cycle
        """
        n_cycles = int(np.ceil(n_frames / self.full_length))
        return n_cycles

    def fit_labels_to_frames(self, n_frames: int) -> list[TimeLabel]:
        """
        Create a list of labels corresponding to each frame in the range of n_frames
        Args:
            n_frames: number of frames to fit labels to
        Returns:
            label_per_frame_list
        """
        n_cycles = self.fit_frames(n_frames)
        label_per_frame_list = np.tile(self.per_frame_list, n_cycles)
        # crop the tail
        return list(label_per_frame_list[0:n_frames])

    def fit_cycles_to_frames(self, n_frames: int) -> list[int]:
        """
        Create a list of integers (what cycle iteration it is) corresponding to each frame in the range of n_frames

        Args:
            n_frames: number of frames to fit cycle iterations to
        Returns:
            cycle_per_frame_list
        """
        n_cycles = self.fit_frames(n_frames)
        cycle_per_frame_list = []
        for n in np.arange(n_cycles):
            cycle_per_frame_list.extend([int(n)] * self.full_length)
        # crop the tail
        return cycle_per_frame_list[0:n_frames]

    def to_dict(self):
        label_order = [label.to_dict() for label in self.label_order]
        d = {'timing': self.duration, 'label_order': label_order}
        return d

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d):
        label_order = [TimeLabel.from_dict(ld) for ld in d['label_order']]
        return cls(label_order, d['timing'])

    @classmethod
    def from_json(cls, j: str):
        """
        j : json string
        """
        d = json.loads(j)
        return cls.from_dict(d)


class Timeline:
    """
    Information about the sequence of labels. Use it when you have non-periodic conditions.
    Args:
        label_order: a list of labels in the right order in which they follow
        duration: duration of the corresponding labels, in frames (based on your imaging). Note that these are
            frames, not volumes !
    """

    def __init__(self, label_order: list[TimeLabel], duration: list[int]):

        # check that all labels are from the same group
        label_group = label_order[0].group
        for label in label_order:
            assert label.group == label_group, \
                f"All labels should be from the same group, but got {label.group} and {label_group}"

        # check that timing is int
        assert all(isinstance(n, int) for n in duration), "duration should be a list of int"

        self.name = label_group
        self.label_order = label_order
        self.duration = list(duration)
        self.full_length = sum(self.duration)
        # list the length of the cycle, each element is the TimeLabel
        # TODO : turn it into an index ?
        self.per_frame_list = self.get_label_per_frame()

    def __eq__(self, other):
        if isinstance(other, Timeline):
            is_same = [
                self.name == other.name,
                self.label_order == other.label_order,
                self.duration == other.duration,
                self.full_length == other.full_length,
                self.per_frame_list == other.per_frame_list
            ]

            return np.all(is_same)
        else:
            print(f"__eq__ is Not Implemented for {Timeline} and {type(other)}")
            return NotImplemented

    def get_label_per_frame(self) -> list[TimeLabel]:
        """
        A list of labels per frame for the duration of the experiment.
        Returns:
            labels per frame for the experiment.
        """
        per_frame_label_list = []
        for (label_time, label) in zip(self.duration, self.label_order):
            per_frame_label_list.extend(label_time * [label])
        return per_frame_label_list

    def __str__(self):
        description = f"Timeline : {self.name}\n"
        description = description + f"Length: {self.full_length}\n"
        for (label_time, label) in zip(self.duration, self.label_order):
            description = description + f"Label {label.name}: for {label_time} frames\n"
        return description

    def __repr__(self):
        return self.__str__()


class FrameManager:
    """
    Deals with frames. Which frames correspond to a volume / cycle/ condition.

    Args:
        file_manager: info about the files.
    """

    def __init__(self, file_manager: FileManager):
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

    def get_frame_mapping(self) -> (list[int], list[int]):
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
    Figures out how to get full volumes for certain time points.

    Learning Resourses:
        maybe I should do type checking automatically, something like here:
        https://stackoverflow.com/questions/9305751/how-to-force-ensure-class-attributes-are-a-specific-type
    Args:
        fpv: frames per volume, number of frames in one volume
        fgf: first good frame, the first frame in the imaging session that is at the top of a volume.
            For example if you started imaging at the top of the volume, fgf = 0,
            but if you started somewhere in the middle, the first good frame is , for example, 23 ...
        frame_manager: the info about the frames
    """

    def __init__(self, fpv: int, frame_manager: FrameManager, fgf: int = 0):

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
        description = description + f"Tailing frames (not a full volume , at the end) : {self.n_tail}\n"
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

    Either frame_to_label_dict or n_frames need to be provided to infer the number of frames.
    If both are provided , they need to agree.

    Args:
        labels: Labels
        info: str, description of the annotation
        frame_to_label: what label it is for each frame
        frame_to_cycle: what cycle it is for each frame
        cycle: for annotation from cycles keeps the cycle
        n_frames: total number of frames, will be inferred from frame_to_label if not provided
    """

    def __init__(self, n_frames: int, labels: Labels, frame_to_label: list[TimeLabel], info: str = None,
                 cycle: Cycle = None, frame_to_cycle: list[int] = None):

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
        assert (frame_to_cycle is None) == (cycle is None), "Creating Annotation: " \
                                                            "You have to provide both cycle and frame_to_cycle."
        # if cycle is provided , check the input and add the info
        if cycle is not None and frame_to_cycle is not None:
            # check that frame_to_cycle is int
            assert all(
                isinstance(n, (int, np.integer)) for n in frame_to_cycle), "frame_to_cycle should be a list of int"
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
            # if one of the annotations has a cycle but the other doesn't
            if (self.cycle is None) != (other.cycle is None):
                return False
            # if both have a cycle, compare cycles as well
            elif self.cycle is not None:
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
    def from_timeline(cls, n_frames, labels, timeline, info=None):
        assert n_frames == timeline.full_length, "number of frames and total timing should be the same"
        # make a fake cycle the length of the whole recording
        frame_to_label = timeline.per_frame_list
        return cls(n_frames, labels, frame_to_label, info=info)

    def get_timeline(self) -> Timeline:
        """
        Transforms frame_to_label to Timeline
        Returns:
            timeline of the resulting annotation
        """
        duration = []
        labels = []
        for label, group in groupby(self.frame_to_label):
            duration.append(sum(1 for _ in group))
            labels.append(label)
        return Timeline(labels, duration)

    def cycle_info(self) -> str:
        """
        Creates and returns a description of a cycle.
        Returns:
            human-readable information about the cycle.
        """
        if self.cycle is None:
            cycle_info = "Annotation doesn't have a cycle"
        else:
            cycle_info = f"{self.cycle.fit_frames(self.n_frames)} full cycles" \
                         f" [{self.n_frames / self.cycle.full_length} exactly]\n"
            cycle_info = cycle_info + self.cycle.__str__()
        return cycle_info

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
    Args:
        db_reader: a DbReader object connected to the database with the experiment description
    """

    def __init__(self, db_reader: DbReader):

        assert isinstance(db_reader, DbReader), "Need DbReader to initialise the Experiment"

        self.db = db_reader
        # will add the loader the first time you are loading anything
        # in load_frames() or load_volumes()
        self.loader = None

    @classmethod
    def create(cls, volume_manager: VolumeManager, annotations: list[Annotation], verbose: bool = False):
        """
        Creates a database instance and initialises the experiment.
        Args:
            volume_manager:
            annotations:
            verbose: whether to print the information about Filemanager, Volumemanager and Annotations on the screen.
        """
        if verbose:
            print(volume_manager.file_manager)
            print(volume_manager)
            for annotation in annotations:
                print(annotation)
                if annotation.cycle is not None:
                    print(annotation.cycle_info())

        db = DbWriter.create()
        db.populate(volumes=volume_manager, annotations=annotations)
        db_reader = DbReader(db.connection)
        return cls(db_reader)

    def save(self, file_name: Union[Path, str]):
        """
        Saves a database into a file.
        Args:
            file_name: full path to a file to save database.
                (Usually the filename would end with .db)
        """
        DbWriter(self.db.connection).save(file_name)

    def add_annotations(self, annotations):
        """
        Adds annotations to existing experiment.
        Does NOT save the changes to disc! run self.save() to save
        """
        DbWriter(self.db.connection).add_annotations(annotations)

    def close(self):
        """
        Close database connection.
        """
        self.db.connection.close()

    @classmethod
    def load(cls, file_name: Union[Path, str]):
        """
        Saves a database into a file.
        Args:
            file_name: full path to a file to load database.
        """
        db_reader = DbReader.load(file_name)
        return cls(db_reader)

    def choose_frames(self, conditions: [tuple], logic: str = "and"):
        """
        Selects the frames that correspond to specified conditions;
        Uses "or" or "and" between the conditions depending on logic.
        To load the selected frames, use load_frames().
        Args:
            conditions: a list of conditions on the annotation labels
                in a form [(group, name),(group, name), ...] where group is a string for the annotation type
                and name is the name of the label of that annotation type. For example [('light', 'on'), ('shape','c')]
            logic: "and" or "or" , default is "and".
        Returns:
            list of frame ids that were chosen. Remember that frame numbers start at 1.
        """
        assert logic == "and" or logic == "or", \
            'between_group_logic should be equal to "and" or "or"'
        frames = []
        if logic == "and":
            frames = self.db.get_and_frames_per_annotations(conditions)
        elif logic == "or":
            frames = self.db.get_or_frames_per_annotations(conditions)

        return frames

    def choose_volumes(self, conditions: Union[tuple, list[tuple]], logic: str = "and", verbose: bool = False):
        """
        Selects only full volumes that correspond to specified conditions;
        Uses "or" or "and" between the conditions depending on logic.
        To load the selected volumes, use load_volumes()
        Args:
            verbose: Whether to print the information about how many frames were chose/ dropped
            conditions: a list of conditions on the annotation labels
                in a form [(group, name),(group, name), ...] where group is a string for the annotation type
                and name is the name of the label of that annotation type. For example [('light', 'on'), ('shape','c')]
            logic: "and" or "or" , default is "and".
        Returns:
            list of volumes and list of frame ids that were chosen.
            Remember that frame numbers start at 1, but volumes start at 0.
        """
        # TODO : make everything start at 1 ????

        assert isinstance(conditions, list) or isinstance(conditions, tuple), f"conditions must be a list or a tuple," \
                                                                              f" but got {type(conditions)} instead"
        if isinstance(conditions, tuple):
            conditions = [conditions]

        # get all the frames that correspond to the conditions
        frames = self.choose_frames(conditions, logic=logic)
        n_frames = len(frames)
        # leave only such frames that correspond to full volumes
        # TODO : why do I even need to return frames?
        volumes, frames = self.db.choose_full_volumes(frames)
        n_dropped = n_frames - len(frames)
        if verbose:
            print(f"Choosing only full volumes. "
                  f"Dropped {n_dropped} frames, kept {len(frames)}")

        return volumes

    def load_volumes(self, volumes: list[int], verbose: bool = False) -> np.ndarray:
        """
        Load volumes. Will load the specified full volumes.
        All the returned volumes or slices should have the same number of frames in them.
        Args:
            volumes: the indexes of volumes to load.
            verbose: Whether to print the information about the loading
        Returns:
            4D array with the loaded volumes.
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
                                               show_progress=verbose)
        return volumes_img

    def list_volumes(self) -> np.ndarray[int]:
        """
        Returns a list of all the volumes IDs in the experiment.
        If partial volumes are present: for "head" returns -1, for "tail" returns -2.
        Returns:
            list of volume IDs
        """
        volume_list = np.array(self.db.get_volume_list())
        if np.sum(volume_list == -1) > 0:
            warnings.warn(f"The are some frames at the beginning of the recording "
                          f"that don't correspond to a full volume.")
        if np.sum(volume_list == -2) > 0:
            warnings.warn(f"The are some frames at the end of the recording "
                          f"that don't correspond to a full volume.")
        return volume_list

    def list_conditions_per_cycle(self, annotation_type: str, as_volumes: bool = True) -> (list[int], list[str]):
        """
        Returns a list of conditions per cycle.
        Args:
            annotation_type: The name of the annotation for which to get the conditions list
            as_volumes: weather to return conditions per frame (default) or per volume.
                If as_volumes is true, it is expected that the conditions are not changing in the middle of the volume.
                Will throw an error if it happens.
        Returns:
            list of the condition ids ( condition per frame or per volume) and corresponding condition names.
        """
        # TODO : check if empty
        if as_volumes:
            _, condition_ids, count = self.db.get_conditionIds_per_cycle_per_volumes(annotation_type)
            fpv = self.db.get_fpv()
            assert np.all(np.array(count) == fpv), "Can't list_conditions_per_cycle with as_volumes=True: " \
                                                   "some conditions don't cover the whole volume." \
                                                   "You might want to get conditions per frame," \
                                                   " by setting as_volumes=False"
        else:
            _, condition_ids = self.db.get_conditionIds_per_cycle_per_frame(annotation_type)
        names = self.db._get_Name_from_AnnotationTypeLabels()

        return condition_ids, names

    def list_cycle_iterations(self, annotation_type: str, as_volumes: bool = True) -> list[int]:
        """
        Returns a list of cycle iteratoins.
        Args:
            annotation_type: The name of the annotation for which to get the cycle iteratoins list
            as_volumes: weather to return cycle iteratoins per frame ( default) or per volume.
                If as_volumes is true, it is expected that the cycle iteratoins are not changing in the middle of the volume.
                Will throw an error if it happens.
            as_volumes: bool
        Returns:
            list of the condition ids (cycle iterations per frame or per volume)
        """
        # TODO : check if empty
        if as_volumes:
            _, cycle_its, count = self.db.get_cycleIterations_per_volumes(annotation_type)
            fpv = self.db.get_fpv()
            assert np.all(np.array(count) == fpv), "Can't list_cycle_iteratoins with as_volumes=True: " \
                                                   "some iteratoins don't cover the whole volume." \
                                                   "You might want to get iteratoins per frame," \
                                                   " by setting as_volumes=False"
        else:
            _, cycle_its = self.db.get_cycleIterations_per_frame(annotation_type)

        return cycle_its

    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

    def summary(self):
        """
        Prints a detailed description of the experiment.
        """
        raise NotImplementedError

"""
Classes to specify the experimental conditions and load necessary data.
"""
from tifffile import TiffFile
import json
import numpy as np
import collections
from pathlib import Path
import glob
from tqdm import tqdm
# import pandas as pd
from sqlite3 import connect


# SAVING AS JSON :
# TODO : Write custom JSONEncoder to make class JSON serializable (???) (or use object_hook)

def to_json(objs, filename):
    """
    Writes an object, or list of objects as json file.
    The objects should have method to_dict()
    """
    if isinstance(objs, list):
        j = json.dumps([obj.to_dict() for obj in objs])
    else:
        j = json.dumps(objs.to_dict())

    with open(filename, 'w') as json_file:
        json_file.write(j)


def from_json(cls, filename):
    """
    Loads an object, or list of objects of class cls from json file.
    The objects should have method from_dict()
    """
    with open(filename) as json_file:
        j = json.load(json_file)

    if isinstance(j, list):
        objs = [cls.from_dict(d) for d in j]
    else:
        objs = cls.from_dict(j)

    return objs


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
        When the timelabel has a group , both names and group need to match.

        :param other: TimeLabel or string to compare to
        :return: True or False , wether the two Timelabels are the same
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

    @classmethod
    def from_dict(cls, d):
        name = d['name']
        description = d['description']
        return cls(name, description)

    def to_dict(self):
        d = {'name': self.name,
             'description': self.description}
        return d


class Labels:
    """
    Describes a particular group of time labels.
    TODO : also responsible for colors for plotting these labels.
    """

    def __init__(self, group, states, state_info={}):
        self.group = group
        self.states = states
        for state in states:
            if state in state_info:
                setattr(self, state, TimeLabel(state, description=state_info[state], group=self.group))
            else:
                setattr(self, state, TimeLabel(state, group=self.group))

    def __str__(self):
        description = f"Label group : {self.group}\n"
        description = description + f"States:\n"
        for state in self.states:
            description = description + f"{getattr(self, state)}\n"
        return description

    def __repr__(self):
        return self.__str__()


class Cycle:
    """
    Information about the repeated cycle of labels. Use it when you have some periodic conditions, like : light
    on , light off, light on, light off... will be made of list of labels [light_on, light_off] that repeat ..."""

    def __init__(self, name, labels, timing):
        """
        :param name: cycle name. Used to identify this specific cycle.
        :type name: str

        :param labels: a list of labels in the right order in which they follow
        :type labels: list[TimeLabel]

        :param timing: timing of the corresponding labels, in frames (based on your imaging). Note that these are
        frames, not volumes !
        :type timing: list[int]
        """
        self.name = name
        self.labels = labels
        self.timing = np.array(timing)
        self.full_length = sum(self.timing)
        # list the length of the cycle, each element is the TimeLabel
        # TODO : turn it into an index ?
        self.per_frame_list = self.get_label_per_frame()

    def get_label_per_frame(self):
        """
        A list of labels per frame for one cycle only.
        """
        per_frame_label_list = []
        for (label_time, label) in zip(self.timing, self.labels):
            per_frame_label_list.extend(label_time * [label])
        return per_frame_label_list

    def __str__(self):
        description = f"Cycle : {self.name}\n"
        description = description + f"Length: {self.full_length}\n"
        for (label_time, label) in zip(self.timing, self.labels):
            description = description + f"Label {label.name}: for {label_time} frames\n"
        return description

    def __repr__(self):
        return self.__str__()

    def fit_frames(self, n_frames):
        """
        Calculates how many cycles you need to fully cover n_frames.
        :param n_frames:
        :return:
        """
        n_cycles = int(np.ceil(n_frames / self.full_length))
        return n_cycles

    def fit_labels_to_frames(self, n_frames):
        """
        Create a list of labels corresponding to each frame in the range of n_frames
        :param n_frames: number of frames to fit labels to
        :return: label_per_frame_list
        """
        n_cycles = self.fit_frames(n_frames)
        label_per_frame_list = np.tile(self.per_frame_list, n_cycles)
        # crop the tail
        return label_per_frame_list[0:n_frames]

    def fit_cycles_to_frames(self, n_frames):
        """
        Create a list of integers (what cycle iteration it is) corresponding to each frame in the range of n_frames
        :param n_frames: number of frames to fit cycle iterations to
        :return: cycle_per_frame_list
        """
        n_cycles = self.fit_frames(n_frames)
        cycle_per_frame_list = []
        for n in np.arange(n_cycles):
            cycle_per_frame_list.extend([n] * self.full_length)
        # crop the tail
        return cycle_per_frame_list[0:n_frames]

    @classmethod
    def from_dict(cls, d):
        """
        Create a cycle from a dictionary. The dictionary can eiter specify labels with a name and description:
        d = {
            "name": "Cycle1_name",
            "labels": [
                        {"name":"label1_name", "description": "label1_description"},
                        {"name":"label2_name", "description": "label2_description"},
                        {"name":"label3_name", "description": "label3_description"}
                       ],
            "timing": [10,15,10]
            }
        Or it can specify them simply with a string (will assigned as labels name):
        d = {
            "name": "Cycle1_name",
            "labels": [label1,label2,label3],
            "timing": [10,15,10]
            }
        """
        # if labels are specified as strings, turn strings into dict
        for idl, dl in enumerate(d['labels']):
            if isinstance(dl, str):
                d['labels'][idl] = {"name": dl, "description": None}

        labels = [TimeLabel.from_dict(dl) for dl in d['labels']]
        timing = np.array(d['timing'])
        name = d['name']
        return cls(name, labels, timing)

    def to_dict(self):
        labels = [label.to_dict() for label in self.labels]
        d = {'timing': self.timing.tolist(), 'labels': labels}
        return d


class FileManager:
    """
    Figures out stuff concerning the many files. For example in what order do stacks go?
    Will grab all the tif files in the provided folder and order them alphabetically.

    :param data_dir: path to the folder with the files, ends with "/" or "\\"
    :type data_dir: str
    """

    def __init__(self, data_dir, file_names=None, frames_in_file=None):

        # 1. get data_dir and check it exists
        self.data_dir = Path(data_dir)
        assert self.data_dir.is_dir(), f"No directory {self.data_dir}"

        # 2. get files
        if file_names is None:
            # if files are not provided , search for tiffs in the data_dir
            self.tif_files = list(self.data_dir.glob("*.tif"))
            self.file_names = [file.name for file in self.tif_files]
        else:
            # if a list of files is provided, check it's in the folder
            self.tif_files = [self.data_dir.joinpath(file) for file in file_names]
            for file in self.tif_files:
                assert file.is_file(), f"File {file} is not found"
            self.file_names = file_names

        # 3. Get number of frames per file
        if frames_in_file is None:
            # if number of frames not provided , search for tiffs in the data_dir
            self.frames_in_file = self.get_frames_in_file()
        else:
            # if provided ... we'll trust you - hope these numbers are correct
            self.frames_in_file = frames_in_file

        self.n_files = len(self.tif_files)

    def change_order(self, order):
        """
        Changes the order of the files. If you notices that files are in the wrong order, provide the new order.
        If you wish to exclude any files, get rid of them ( don't include their IDs into the new order ).

        :param order: The new order in which the files follow. Refer to file by it's position in the original list.
        Should be the same length as the number of files in the original list, or smaller (if you want to get rid of
        some files).
        :type order: list[int]
        """
        assert len(np.unique(order)) > self.n_files, \
            "Number of unique files is smaller than elements in the list! "

        self.tif_files = [self.tif_files[i] for i in order]
        self.frames_in_file = [self.frames_in_file[i] for i in order]

    def get_frames_in_file(self):
        """
        Get the number of frames  per file.
        returns a list with number fof frames per file.
        Expand this method if you want to work with other file types (not tiffs).
        """
        frames_in_file = []
        for tif_file in self.tif_files:
            # setting multifile to false since sometimes there is a problem with the corrupted metadata
            # not using metadate, since for some files it is corrupted for unknown reason ...
            stack = TiffFile(tif_file, _multifile=False)
            n_frames = len(stack.pages)
            frames_in_file.append(n_frames)
            stack.close()
        return frames_in_file

    def __str__(self):
        description = f"Total of {self.n_files} files.\nCheck the order :\n"
        for i_file, file in enumerate(self.file_names):
            description = description + "[ " + str(i_file) + " ] " + file + " : " + str(
                self.frames_in_file[i_file]) + " frames\n"
        return description

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_dict(cls, d):
        """
        Initialises FileManager object from dictionary.
        """
        project_dir = d['project_dir']
        tif_files = d['tif_files']
        frames_in_file = d['frames_in_file']
        return cls(project_dir, file_names=tif_files, frames_in_file=frames_in_file)

    def to_dict(self):
        """
        Writes FileManager object to dictionary.
        """
        d = {'project_dir': self.data_dir,
             'tif_files': self.tif_files,
             'frames_in_file': self.frames_in_file}
        return d


class FrameManager:
    """
    Deals with frames. Which frames correspond to a volume / cycle/ condition.

    :param file_manager: info about the files.
    :type file_manager: FileManager
    """

    def __init__(self, file_manager, n_frames=None, frame_size=None, frame_to_file=None):
        self.file_manager = file_manager
        # TODO : create separate method "from_filemanager" and get all the None cases there ...
        if n_frames is None:
            self.n_frames = self.get_n_frames()
        else:
            self.n_frames = n_frames

        if frame_size is None:
            self.frame_size = self.get_frame_size()
        else:
            self.frame_size = frame_size

        if frame_to_file is None:
            self.frame_to_file = self.get_frame_list()
        else:
            self.frame_to_file = frame_to_file

    def get_n_frames(self):
        """
        Returns total number of frames (all files from file_manager combined).

        :return: Frames per file.
        :rtype: int
        """
        n_frames = np.sum(self.file_manager.frames_in_file)
        return n_frames

    def get_frame_list(self):
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
        frame_to_file = {'file_idx': [],
                         'in_file_frame': []}  # collections.defaultdict(list)

        for file_idx in range(self.file_manager.n_files):
            n_frames = self.file_manager.frames_in_file[file_idx]
            frame_to_file['file_idx'].extend(n_frames * [file_idx])
            frame_to_file['in_file_frame'].extend(np.arange(n_frames).tolist())

        frame_to_file['file_idx'] = frame_to_file['file_idx']
        frame_to_file['in_file_frame'] = frame_to_file['in_file_frame']

        return frame_to_file

    def get_frame_to_file_name(self):
        """
        Returns a list : for each frame , the filename to which that frame belongs.
        """
        return [self.file_manager.tif_files[idx] for idx in self.frame_to_file['file_idx']]

    def get_frame_size(self):
        """
        Gets frame size ( height , width ).
        
        :return: height and width of an individual frame in pixels
        """
        # initialise tif file and open the stack
        tif_file = self.file_manager.tif_files[0]
        stack = TiffFile(tif_file, _multifile=False)
        page = stack.pages.get(0)
        h, w = page.shape
        stack.close()
        return h, w

    def load_frames(self, frames, report_files=False, show_progress=True):
        """
        Load frames from files and return as an array (frame, y, x).

        :param frames: list of frames to load
        :type frames: list[int]

        :param report_files: whether to print the file from which the frames are loaded on the screen.
        :type report_files: bool

        :param show_progress: whether to show the progress bar of how many frames have been loaded.
        :type show_progress: bool

        :return: 3D array of requested frames (frame, y, x)
        :rtype: numpy.ndarray
        """
        if report_files:
            show_progress = False
            print("Set show_progress to False , because report_files  is set to True")

        # prepare an empty array:
        h, w = self.frame_size
        frames_img = np.zeros((len(frames), h, w))

        # initialise tif file and open the stack
        tif_idx = self.frame_to_file['file_idx'][frames[0]]
        tif_file = self.file_manager.tif_files[tif_idx]
        stack = TiffFile(tif_file, _multifile=False)

        if report_files:
            print(f'Loading from file {tif_idx}')
        for i, frame in enumerate(tqdm(frames, disable=not show_progress)):
            # locate frame in file
            frame_idx = self.frame_to_file['in_file_frame'][frame]
            # check if he frame belongs to an opened file
            if tif_idx == self.frame_to_file['file_idx'][frame]:
                # continue loading from the previously opened file
                frames_img[i, :, :] = stack.asarray(frame_idx)
            else:
                # switch to a different file
                tif_idx = self.frame_to_file['file_idx'][frame]
                tif_file = self.file_manager.tif_files[tif_idx]
                if report_files:
                    print(f'Loading from file {tif_idx}')
                stack.close()
                stack = TiffFile(tif_file, _multifile=False)
                frames_img[i, :, :] = stack.asarray(frame_idx)
        stack.close()
        return frames_img

    def __str__(self):
        return f"Total {self.n_frames} frames."

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_dict(cls, d):
        file_manager = FileManager.from_dict(d['file_manager'])
        n_frames = d['n_frames']
        frame_size = d['frame_size']
        frame_to_file = d['frame_to_file']
        return cls(file_manager, n_frames=n_frames, frame_size=frame_size, frame_to_file=frame_to_file)

    def to_dict(self):
        d = {'file_manager': self.file_manager.to_dict(),
             'n_frames': int(self.n_frames),
             'frame_size': self.frame_size,
             'frame_to_file': self.frame_to_file}
        return d


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
        # frames per volume
        self.fpv = fpv
        # first good frame, start counting from 0 : 0, 1, 2, 3, ...
        self.fgf = fgf
        # total number of frames
        self.frame_manager = frame_manager
        self.n_frames = frame_manager.n_frames
        # frames at the beginning, full volumes and frames at the end
        self.n_head = self.fgf
        self.full_volumes, self.n_tail = divmod((self.n_frames - self.fgf), self.fpv)
        # frames to slices :
        self.frame_to_z = self.get_frames_to_z()
        # frames to full volumes :
        self.frame_to_vol = self.get_frames_to_volumes()

    @classmethod
    def from_dict(cls, d):
        fpv = d['fpv']
        fgf = d['fgf']
        frame_manager = FrameManager.from_dict(d['frame_manager'])
        return cls(fpv, frame_manager, fgf=fgf)

    def to_dict(self):
        d = {'frame_manager': self.frame_manager.to_dict(),
             'fpv': self.fpv,
             'fgf': self.fgf}
        return d

    def __str__(self):
        description = ""
        description = description + f"Total frames : {self.n_frames}\n"
        description = description + f"Volumes start on frame : {self.fgf}\n"
        description = description + f"Total good volumes : {self.full_volumes}\n"
        description = description + f"Frames per volume : {self.fpv}\n"
        return description

    def __repr__(self):
        return self.__str__()

    def get_frames_to_z(self):
        z_per_frame_list = np.arange(self.fpv)
        # set at what z the imaging starts and ends
        i_from = self.fpv - self.n_head
        i_to = self.n_tail - self.fpv
        # map frames to z
        frame_to_z = np.tile(z_per_frame_list, self.full_volumes + 2)[i_from:i_to]
        return frame_to_z

    def get_frames_to_volumes(self):
        """
        maps frames to volumes
        -1 for head ( not full volume at the beginning )
        volume number for full volumes : 0, 1, ,2 3, ...
        -2 for tail (not full volume at the end )
        """
        frame_to_vol = [-1] * self.n_head
        for vol in np.arange(self.full_volumes):
            frame_to_vol.extend([vol] * self.fpv)
        frame_to_vol.extend([-2] * self.n_tail)
        return np.array(frame_to_vol)

    def load_volumes(self, volumes, verbose=False, show_progress=True):
        """
        Loads specified volumes.
        :param volumes: list of volumes indices to load.
        :type volumes: list[int]

        :param verbose: whether to print the file from which the frames are loaded on the screen.
        :type verbose: bool

        :param show_progress: whether to show the progress bar of how many frames have been loaded.
        :type show_progress: bool

        :return: 4D array of shape (number of volumes, zslices, height, width)
        :rtype: numpy.ndarray
        """
        w, h = self.frame_manager.get_frame_size()
        # convert volumes to frames: get all the frames corresponding to the requested volumes
        which_frames = np.where(np.in1d(self.frame_to_vol, volumes))[0]
        # load frames and shape them into volumes
        frames_reshaped = self.frame_manager.load_frames(which_frames,
                                                         report_files=verbose,
                                                         show_progress=show_progress).reshape(
            (len(volumes), self.fpv, w, h))
        return frames_reshaped

    def load_zslices(self, zpos, slices=None):
        """
        Loads specified zslices.
        For example only the first 3 z slices at z 30 : zpos = 30, slices= [0,1,2];
        all z slices at z 30 : zpos = 30, slices = None

        :param zpos: z position, what z slice to load.
        :type zpos: int

        :param slices: which of the slices to load. If None loads all.
        :type slices: int or list[int] or numpy.ndarray
        # TODO : writing stuff like list[int] or numpy.ndarray seems wrong, verify ...

        :return: 3D array of shape (number of zslices, height, width)
        :rtype: numpy.ndarray
        """
        # convert z slices to frames
        which_frames = np.where(np.in1d(self.frame_to_z, zpos))[0]
        if slices:
            which_frames = which_frames[slices]

        frames = self.frame_manager.load_frames(which_frames)

        return frames


class Annotation:
    """
    Time annotation of the experiment.
    """

    def __init__(self,
                 frame_to_label_dict=None,
                 frame_to_cycle_dict=None,
                 cycles=None,
                 n_frames=None):
        """
        Either frame_to_label_dict or n_frames need to be provided to infer the number of frames.
        If both are provided , they need to agree.
        :param frame_to_label: dictionaly {annotation group : frame_to_label_list}
        :param frame_to_cycle_dict: dictionaly {annotation group : frame_to_cycle_list}
        :param cycles: list of cycles
        :param n_frames: total number of frames, will be inferred from frame_to_label if not provided
        """
        # 1. get total experiment length in frames
        if frame_to_label_dict is not None:
            self.n_frames = len(frame_to_label_dict.values()[0])
            # check that all mappings in frame_to_label_dict are the same length
            for key, value in frame_to_label_dict.items():
                assert len(value) == self.n_frames, f"The number of frames in the new annotation for {key}: " \
                                                    f" {len(value)}, " \
                                                    f"and the number of frames in the existing annotations ," \
                                                    f" {self.n_frames}, do not match."
            # check that the n_frames provided ( if any ) and number of frames inferred are the same
            if n_frames is not None:
                assert n_frames == self.n_frames, f"The number of frames given as input, {n_frames}, " \
                                                  f"and the number of frames in the provided annotations ," \
                                                  f" {self.n_frames}, do not match."
        elif n_frames is not None:
            self.n_frames = n_frames
        else:
            raise ValueError("Either n_frames or frame_to_label_dict need to be provided. ")

        if frame_to_cycle_dict is not None:
            # check that all mappings in frame_to_label_dict are the same length
            for key, value in frame_to_cycle_dict.items():
                assert len(value) == self.n_frames, f"The number of frames in the frame_to_cycle mapping for {key}: " \
                                                    f" {len(value)}, " \
                                                    f"and the number of frames in the existing annotations ," \
                                                    f" {self.n_frames}, do not match."
        # 2. assign frame_to_label and frame_to_cycle
        self.frame_to_label = frame_to_label_dict
        self.frame_to_cycle = frame_to_cycle_dict

        # 3. add information from cycles to frame_to_label and frame_to_cycle
        if cycles is not None:
            frame_to_label = self.frame_to_label_from_cycles(cycles)
            frame_to_cycle = self.frame_to_cycle_from_cycles(cycles)

            if self.frame_to_label is None:
                self.frame_to_label = frame_to_label
            else:
                self.add_frame_to_label(frame_to_label)

            if self.frame_to_cycle is None:
                self.frame_to_cycle = frame_to_cycle
            else:
                self.add_frame_to_cycle(frame_to_cycle)

    def __str__(self):
        description = ""
        description = description + f"Total frames : {self.n_frames}\n"
        for cycle in self.cycles:
            description = description + cycle.__str__()
        return description

    def __repr__(self):
        return self.__str__()

    # adding to annotation methods _____________________________________________________________________________
    def add_frame_to_cycle(self, frame_to_cycle):
        """
        Appends information about additional label groups to frame_to_cycle dictionary

        :param frame_to_cycle: list of labels for every frame
        :return: None
        """
        for key, value in frame_to_cycle.items():
            assert key not in self.frame_to_cycle, f"{key} cycle group is already in the annotation." \
                                                   f" If you are trying to add a new group, rename it. " \
                                                   f"To rewrite an existing group, first delete it and then add."

            assert len(value) == self.n_frames, f"The number of frames in the new annotation, {len(value)}, " \
                                                f"and the number of frames in the existing annotations ," \
                                                f" {self.n_frames}, do not match."

        self.frame_to_cycle.update(frame_to_cycle)

    def add_frame_to_label(self, frame_to_label):
        """
        Appends information about additional label groups to frame_to_label dictionary

        :param frame_to_label: list of labels for every frame
        :return: None
        """
        for key, value in frame_to_label.items():
            assert key not in self.frame_to_label, f"{key} label group is already in the annotation." \
                                                   f" If you are trying to add a new group, rename it. " \
                                                   f"To rewrite an existing group, first delete it and then add."

            assert len(value) == self.n_frames, f"The number of frames in the new annotation, {len(value)}, " \
                                                f"and the number of frames in the existing annotations ," \
                                                f" {self.n_frames}, do not match."

        self.frame_to_label.update(frame_to_label)

    # cycles related methods ________________________________________________________________________
    def frame_to_label_from_cycles(self, cycles):
        """
        Maps frames to conditions for each label group.

        :return: for each label group, a list of length equal to the total number of frames in all the files,
        where each element corresponds to a frame and contains the condition presented during that frame.

        :rtype: dict(str : list[TimeLabel])
        """
        d = {}
        for cycle in cycles:
            d[cycle.name] = cycle.fit_labels_to_frames(self.n_frames)
        return d

    def frame_to_cycle_from_cycles(self, cycles):
        """
        Maps frames to cycles for all cycle groups.
        Gives you approximately this information :
         " For cycle with cycle.name = " shape" , the first 10 labels correspond to cycle #1 ,
          the next 10 frames to cycle #2 etc etc"

        :return: dict{cycle name : cycle per frame list}
        """
        # TODO : add the cycle start
        # frames before the cycle starts are labeled with -1
        # cycle_per_frame_list = [-1] * self.cycle_start
        d = {}
        for cycle in cycles:
            d[cycle.name] = cycle.fit_cycles_to_frames(self.n_frames)
        return d

    # to/from dict methods ________________________________________________________________________
    @classmethod
    def from_dict(cls, d):
        frame_to_label_dict = d['frame_to_label']
        frame_to_label_dict = d['frame_to_label']
        n_frames = d['n_frames']
        return cls(frame_to_label_dict=frame_to_label_dict, n_frames=n_frames)

    def to_dict(self):
        d = {'frame_to_label': self.frame_to_label,
             'n_frames': self.n_frames}
        return d

class Experiment:
    """
    Information about the experiment. Can contain cycles and more.
    For now assumes there are always repeating cycles, but if you have only one repetition it will still work fine ;)
    If you don't need to track the conditions and only need to track volumes/ z-slices,
    """

    def __init__(self, frame_manager, volume_manager, annotation, cycles=None):
        """
        :param volume_manager: volume manager
        :param annotation: annotation
        :param cycles: cycles ( optional - if you want to keep cycles information )
        """

        self.volume_manager = volume_manager
        self.frame_manager = volume_manager.frame_manager
        self.annotation = annotation
        assert self.frame_manager.n_frames == self.annotation.n_frames, \
            f"Number of frames in the frame manager, {self.frame_manager.n_frames}, " \
            f"doesn't match the number of frames in the annotation , {self.annotation.n_frames}"
        self.cycles = cycles

        # built database
        # self.info_df = self.frame_manager.create_info()
        # self.info_df = self.volume_manager.append_info(self.info_df)
        # self.info_df = self.annotation.append_info(self.info_df)

    @classmethod
    def from_dict(cls, d):
        volume_manager = VolumeManager.from_dict(d['volume_manager'])
        annotation = Annotation.from_dict(d['annotation'])
        cycles = []
        for cycle in d['cycles']:
            cycles.append(Cycle.from_dict(cycle))
        return cls(volume_manager, annotation, cycles=cycles)

    def to_dict(self):
        if self.cycles is not None:
            cycles = []
            for cycle in self.cycles:
                cycles.append(cycle.to_dict())
        else:
            cycles = None

        d = {'volume_manager': self.volume_manager.to_dict(),
             'annotation': self.annotation.to_dict(),
             'cycles': cycles}
        return d

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

    @classmethod
    def from_tables(cls, spec):
        """
        Initialise experiment object from tables.
        """
        # prepare frame manager
        frame_manager = FrameManager.from_csv(spec['frame_manager'])
        # TODO : why do I need frame manager in volume manager?
        volume_manager = VolumeManager.from_csv(spec['volume_manager'], frame_manager)
        annotation = Annotation.from_csv(spec['annotation'])
        # cycles = ???

        #
        # # prepare annotation
        # cycles = []
        # for k in spec['annotation']:
        #     spec['annotation'][k]['name'] = k
        #     cycles.append(Cycle.from_dict(spec['annotation'][k]))
        # annotation = Annotation(frame_manager, cycles)
        raise NotImplementedError

        return cls(volume_manager, annotation)


class DbManager:
    """
    Deals with the database methods
    Code adopted from https://www.devdungeon.com/content/python-sqlite3-tutorial
    Maybe better stuff here : https://www.sqlitetutorial.net/sqlite-python/creating-tables/
    """

    def __init__(self, db):
        self.db_connection = db

    def create_tables(self):
        db_cursor = self.db_connection.cursor()

        sql_create_Options_table = """
            CREATE TABLE "Options" (
            "Key"	TEXT NOT NULL UNIQUE,
            "Value"	TEXT NOT NULL,
             PRIMARY KEY("Key")
            )
            """
        db_cursor.execute(sql_create_Options_table)

        sql_create_Files_table = """
            CREATE TABLE "Files" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "FileName"	TEXT NOT NULL UNIQUE,
            "NumFrames"	INTEGER NOT NULL,
            PRIMARY KEY("Id" AUTOINCREMENT)
            )
            """
        db_cursor.execute(sql_create_Files_table)

        sql_create_AnnotationTypes_table = """
            CREATE TABLE "AnnotationTypes" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "Name"	TEXT NOT NULL UNIQUE,
            "Description"	TEXT,
            PRIMARY KEY("Id" AUTOINCREMENT)
            )
            """
        db_cursor.execute(sql_create_AnnotationTypes_table)

        sql_create_Frames_table = """
            CREATE TABLE "Frames" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "FrameInFile"	INTEGER NOT NULL,
            "FileId"	INTEGER NOT NULL,
            PRIMARY KEY("Id" AUTOINCREMENT),
            FOREIGN KEY("FileId") REFERENCES "Files"("Id")
            )
            """
        db_cursor.execute(sql_create_Frames_table)

        sql_create_Cycles_table = """
            CREATE TABLE "Cycles" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "AnnotationTypeId"	INTEGER NOT NULL UNIQUE,
            "Structure"	TEXT NOT NULL,
            FOREIGN KEY("AnnotationTypeId") REFERENCES "AnnotationTypes"("Id"),
            PRIMARY KEY("Id" AUTOINCREMENT)
            )
            """
        db_cursor.execute(sql_create_Cycles_table)

        sql_create_AnnotationTypeLabels_table = """
            CREATE TABLE "AnnotationTypeLabels" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "AnnotationTypeId "	INTEGER NOT NULL,
            "Name"	TEXT NOT NULL,
            "Description"	TEXT,
            PRIMARY KEY("Id" AUTOINCREMENT),
            FOREIGN KEY("AnnotationTypeId ") REFERENCES "AnnotationTypes"("Id"),
            UNIQUE("AnnotationTypeId ","Name")
            )
            """
        db_cursor.execute(sql_create_AnnotationTypeLabels_table)

        sql_create_Annotations_table = """
            CREATE TABLE "Annotations" (
            "FrameId"	INTEGER NOT NULL UNIQUE,
            "AnnotationTypeLabelId"	INTEGER NOT NULL,
            FOREIGN KEY("FrameId") REFERENCES "Frames"("Id"),
            FOREIGN KEY("AnnotationTypeLabelId") REFERENCES "AnnotationTypeLabels"("Id"),
            PRIMARY KEY("FrameId" AUTOINCREMENT)
            )
            """
        db_cursor.execute(sql_create_Annotations_table)

        sql_create_CycleIterations_table = """
            CREATE TABLE "CycleIterations" (
            "FrameId"	INTEGER NOT NULL,
            "CycleId"	INTEGER NOT NULL,
            "CycleIteration"	INTEGER NOT NULL,
            FOREIGN KEY("CycleId") REFERENCES "Cycles"("Id"),
            FOREIGN KEY("FrameId") REFERENCES "Frames"("Id")
            )
            """
        db_cursor.execute(sql_create_CycleIterations_table)

        sql_create_Volumes_table = """
            CREATE TABLE "Volumes" (
            "FrameId"	INTEGER NOT NULL UNIQUE,
            "VolumeId"	INTEGER NOT NULL,
            "SliceInVolume"	INTEGER NOT NULL,
            PRIMARY KEY("FrameId" AUTOINCREMENT),
            FOREIGN KEY("FrameId") REFERENCES "Frames"("Id")
            )
            """
        db_cursor.execute(sql_create_Volumes_table)

        db_cursor.close()

    def populate_Options(self, d):
        """
        d: dictionary with key - value pairs
        """
        pass

    def populate_Files(self, file_name, num_frames):
        """
        file_name : list with filenames per file (str)
        num_frames : list with number of frames per file (int)
        """
        pass

    def populate_AnnotationTypes(self, name, description):
        """
        """
        pass

    def populate_Frames(self, file_name, frame_in_file):
        """
        Something like :
        insert into tab2 (id_customers, value)
        values ((select id from tab1 where customers='john'), 'alfa');
        but in SQLite
        https://www.tutorialspoint.com/sqlite/sqlite_insert_query.htm
        """
        pass

    def populate_Cycles(self, name, structure):
        """
        search for AnnotationTypeId by name
        """
        pass

    def populate_AnnotationTypeLabels(self):
        pass

    def populate_Annotations(self):
        pass

    def populate_CycleIterations(self):
        pass

    def populate_Volumes(self):
        pass

    @staticmethod
    def save_db(file_name, memory_db):
        """
        Backup a memory database to a file
        file_name : "databasename.db"
        """
        backup_db = connect(file_name)
        memory_db.backup(backup_db)
        memory_db.close()
        backup_db.close()

    @classmethod
    def initialise_db(cls):
        """
        Creates an empty DB to the experiment in memory
        """
        # For an in-memory only database:
        memory_db = connect(':memory:')
        # ....
        # TODO : fill out the database
        return cls(memory_db)

    @classmethod
    def load_db(cls, file_name):
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

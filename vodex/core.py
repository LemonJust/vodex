"""
Classes to specify the experimental conditions and load necessary data.
"""
from tifffile import TiffFile
import json
import numpy as np
import collections
import glob
from tqdm import tqdm


# SAVING AS JSON :
# TODO : Write custom JSONEncoder to make class JSON serializable (???)

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


class Stimulus:
    """
    Describes a particular stimulus presented during the experiment.
    Any specific aspect of the experiment is a stimulus : temperature|light|sound|image on the screen|drug ... etc.

    :param name: the name for the stimulus. This is a unique identifier of the stimulus.
                    Different stimuli should have different names.
                    Different Stimuli are compared based on their names, so same name means it is the same stimuli.
    :type name: str

    :param description: a detailed description of the stimulus. This is to give you more info, but it is not used for
    anything else. Make sure that different stmuli have different name, because they are compared by name,
    not by description.
    :type description: str
    """

    def __init__(self, name, description):
        self.name = name
        self.description = description

    def __str__(self):
        return f"Stimulus {self.name}: {self.description}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        # comparing by name
        if not isinstance(other, Stimulus):
            return NotImplemented
        return self.name == other.name

    def __ne__(self, other):
        # comparing by name
        if not isinstance(other, Stimulus):
            return NotImplemented
        return not self.name == other.name

    @classmethod
    def from_dict(cls, d):
        name = d['name']
        description = d['description']
        return cls(name, description)

    def to_dict(self):
        d = {'name': self.name, 'description': self.description}
        return d


class Condition:
    """
    Information about the experiment conditions. Condition can be made of multiple simultaneous stimuli and
    conditions are compared based on the stimuli that that contain. Note that their order doesn't matter.

    :param stimuli: a list of stimuli that make up the condition
    :type stimuli: Stimulus or list[Stimulus]
    """

    def __init__(self, stimuli, name=None):
        if not isinstance(stimuli, list):
            stimuli = [stimuli]
        # TODO : figure out why
        #  if not isinstance(stimuli, Stimulus):
        #       ...
        #  else:
        #       raise AssertionError(f"Input
        #  should be Stimulus or list[Stimulus], not {type(stimuli)}") doesn't work and gives <class 'core.Stimulus'>
        self.stimuli = stimuli
        self.names = [stimulus.name for stimulus in stimuli]
        self.name = name

    def __str__(self):
        description = ""
        if self.name is not None:
            description = description + f"Condition {self.name}:\n"
        else:
            description = description + f"Condition:\n"
        for stimulus in self.stimuli:
            description = description + f"Stimulus {stimulus.name}: {stimulus.description}\n"
        return description

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Condition):
            return NotImplemented
        # comparing by names, ignoring order
        return set(self.names) == set(other.names)

    def __ne__(self, other):
        if not isinstance(other, Condition):
            return NotImplemented
        # comparing by names, ignoring order
        return not set(self.names) == set(other.names)

    @classmethod
    def from_dict(cls, d):
        stimuli = [Stimulus.from_dict(ds) for ds in d['stimuli']]
        if 'name' in d:
            name = d['name']
        else:
            name = None
        return cls(stimuli, name=name)

    def to_dict(self):
        stimuli = [stimulus.to_dict() for stimulus in self.stimuli]
        d = {'name': self.name, 'stimuli': stimuli}
        return d


class Cycle:
    """
    Information about the repeated cycle of conditions. Use it when you have some periodic conditions, like : light
    on , light off, light on, light off... will be made of list of conditions [light_on, light_off] that repeat ...

    :param conditions: a list of conditions in the right order in which conditions follow
    :type conditions: list[Condition]

    :param timing: timing of the corresponding conditions, in frames (based on your imaging). Note that these are
    frames, not volumes !
    :type timing: list[int]
    """

    def __init__(self, conditions, timing):

        self.conditions = conditions
        self.timing = timing
        self.full_length = sum(self.timing)
        # list the length of the cycle, each element is the condition ( index as in the conditions list )
        self.per_frame_list = self.get_per_frame_list()

    def get_per_frame_list(self):
        per_frame_condition_list = []
        for (condition_time, condition) in zip(self.timing, self.conditions):
            per_frame_condition_list.extend(condition_time * [condition])
        return per_frame_condition_list

    def __str__(self):
        description = f"Cycle length: {self.full_length}\n"
        for (condition_time, condition) in zip(self.timing, self.conditions):
            description = description + f"Condition {condition.names}: for {condition_time} frames\n"
        return description

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_dict(cls, d):
        conditions = [Condition.from_dict(ds) for ds in d['conditions']]
        timing = d['timing']
        return cls(conditions, timing)

    def to_dict(self):
        conditions = [condition.to_dict() for condition in self.conditions]
        d = {'timing': self.timing, 'conditions': conditions}
        return d


class FileManager:
    """
    Figures out stuff concerning the many files. For example in what order do stacks go?
    Will grab all the tif files in the provided folder and order them alphabetically.

    :param project_dir: path to the folder with the files, ends with "/" or "\\"
    :type project_dir: str
    """

    def __init__(self, project_dir, tif_files=None, frames_in_file=None):
        self.project_dir = project_dir

        if tif_files is None:
            self.tif_files = [file for file in glob.glob(f"{project_dir}*.tif")]
        else:
            self.tif_files = tif_files

        self.n_files = len(self.tif_files)

        if frames_in_file is None:
            self.frames_in_file = self.get_n_frames()
        else:
            self.frames_in_file = frames_in_file

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
            "Number of unique files is smaller than in the list! "

        self.tif_files = [self.tif_files[i] for i in order]
        self.frames_in_file = [self.frames_in_file[i] for i in order]

    def get_n_frames(self):
        """
        Get the number of frames  per file.
        """
        frames_in_file = []
        for tif_file in self.tif_files:
            # setting multifile to false since sometimes there was a problem with the corrupted metadata
            stack = TiffFile(tif_file, _multifile=False)
            n_frames = len(stack.pages)
            frames_in_file.append(n_frames)
            stack.close()
        return frames_in_file

    def __str__(self):
        description = f"Total of {self.n_files} files.\nCheck the order :\n"
        for i_file, file in enumerate(self.tif_files):
            description = description + "[ " + str(i_file) + " ] " + file.split('\\')[-1] + " : " + str(
                self.frames_in_file[i_file]) + " frames\n"
        return description

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_dict(cls, d):
        project_dir = d['project_dir']
        tif_files = d['tif_files']
        frames_in_file = d['frames_in_file']
        return cls(project_dir, tif_files=tif_files, frames_in_file=frames_in_file)

    def to_dict(self):
        d = {'project_dir': self.project_dir,
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
        Collect info about the number of frames per file.

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
        frame_to_file = collections.defaultdict(list)
        for file_idx in range(self.file_manager.n_files):
            n_frames = self.file_manager.frames_in_file[file_idx]
            frame_to_file['file_idx'].extend(n_frames * [file_idx])
            frame_to_file['in_file_frame'].extend(np.arange(n_frames))
        return frame_to_file

    def get_frame_size(self):
        """
        Gets frame size.
        
        :return: height and width of an individual frame in pixels
        """
        # initialise tif file and open the stack
        tif_file = self.file_manager.tif_files[0]
        stack = TiffFile(tif_file, _multifile=False)
        page = stack.pages.get(0)
        h, w = page.shape
        stack.close()
        return h, w

    def load_frames(self, frames, verbose=False, show_progress=True):
        """
        Load frames from files and return as an array.

        :param frames: list of frames to load
        :type frames: list[int]

        :param verbose: whether to print the file from which the frames are loaded on the screen.
        :type verbose: bool

        :param show_progress: whether to show the progress bar of how many frames have been loaded.
        :type show_progress: bool

        :return: 3D array of requested frames (frame, y, x)
        :rtype: numpy.ndarray
        """
        if verbose:
            show_progress = False

        # prepare an empty array, for this grab the size w, h:
        h, w = self.frame_size
        frames_img = np.zeros((len(frames), h, w))

        # initialise tif file and open the stack
        tif_idx = self.frame_to_file['file_idx'][frames[0]]
        tif_file = self.file_manager.tif_files[tif_idx]
        stack = TiffFile(tif_file, _multifile=False)
        # if verbose:
        #     print(f'Loading from file {tif_idx}')
        for i, frame in enumerate(tqdm(frames, disable=not show_progress)):
            frame_idx = self.frame_to_file['in_file_frame'][frame]
            if tif_idx == self.frame_to_file['file_idx'][frame]:
                frames_img[i, :, :] = stack.asarray(frame_idx)
            else:
                stack.close()
                # update the file we are working with
                tif_idx = self.frame_to_file['file_idx'][frame]
                if verbose:
                    print(f'Loading from file {tif_idx}')
                tif_file = self.file_manager.tif_files[tif_idx]
                # open the stack
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
             'n_frames': self.n_frames,
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
        # frames at the beginning, full volumes and at the end
        self.n_head = self.fgf
        self.full_volumes, self.n_tail = divmod((self.n_frames - self.fgf), self.fpv)
        # frames to z :
        self.frame_to_z = self.get_frames_to_z_list()
        self.frame_to_vol = self.get_frames_to_volumes_list()

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
        description = description + f"Total good volumes : {self.full_volumes}\n"
        description = description + f"Frames per volume : {self.fpv}\n"
        return description

    def get_frames_to_z_list(self):
        cycle_per_frame_list = np.arange(self.fpv)
        i_from = self.fpv - self.n_head
        i_to = self.n_tail - self.fpv
        frame_to_z = np.tile(cycle_per_frame_list, self.full_volumes + 2)[i_from:i_to]
        return frame_to_z

    def get_frames_to_volumes_list(self):
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
        which_frames = np.where(np.in1d(self.frame_to_vol, volumes))[0]
        frames_reshaped = self.frame_manager.load_frames(which_frames,
                                                         verbose=verbose, show_progress=show_progress).reshape(
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
        # TODO : list[int] or numpy.ndarray seems wrong, verify ...

        :return: 3D array of shape (number of zslices, height, width)
        :rtype: numpy.ndarray
        """

        which_frames = np.where(np.in1d(self.frame_to_z, zpos))[0]
        if slices:
            which_frames = which_frames[slices]

        frames = self.frame_manager.load_frames(which_frames)

        return frames


class Experiment:
    """
    Information about the experiment. Can contain cycles and more.
    For now assumes there are always repeating cycles, but if you have only one repetition it will still work fine ;)
    If you don't need to track the conditions and only need to track volumes/ z-slices,
    """

    def __init__(self, frame_manager, volume_manager, cycle):

        self.cycle = cycle
        # TODO : frame manager doesn't need to be separate here, since it's already in volume manager ... change it!
        self.frame_manager = frame_manager
        self.volume_manager = volume_manager
        # total experiment length in frames
        self.n_frames = frame_manager.n_frames
        self.full_cycles = int(np.ceil(self.n_frames / self.cycle.full_length))
        self.frame_to_condition = self.get_frame_to_condition_list()

    @classmethod
    def from_dict(cls, d):
        frame_manager = FrameManager.from_dict(d['frame_manager'])
        volume_manager = VolumeManager.from_dict(d['volume_manager'])
        cycle = Cycle.from_dict(d['cycle'])
        return cls(frame_manager, volume_manager, cycle)

    def to_dict(self):
        d = {'frame_manager': self.frame_manager.to_dict(),
             'volume_manager': self.volume_manager.to_dict(),
             'cycle': self.cycle.to_dic()}
        return d

    def get_frame_to_condition_list(self):
        """
        Creates a mapping of frames to conditions.

        :return: a list of length equal to the total number of frames in all the files,
        where each element corresponds to a frame and contains the condition presented during that frame.

        :rtype: list[Condition]
        """
        frame_to_condition_list = np.tile(self.cycle.per_frame_list, self.full_cycles)[0:self.n_frames]
        return frame_to_condition_list

    def select_zslices(self, zslice, condition=None):
        """
        Selects the frames for a specific zslice that correspond to a specified condition.
        If condition is None, selects all the frames for the specified zslice.
        To actually load the selected frames, use frame_manager.load_frames()

        :param zslice: the zslice for which to select the frames
        :type zslice: int

        :param condition: the condition for which to select the frames, or None
        :type condition: Condition

        :return: 1D array of indices: a list of frames corresponding to the zslice and the condition.
        :rtype: numpy.ndarray
        """
        zslice_match = self.volume_manager.frame_to_z == zslice

        if condition is not None:
            condition_match = self.frame_to_condition == condition
            request_met = np.logical_and(condition_match, zslice_match)
        else:
            request_met = zslice_match
        which_frames = np.where(request_met)[0]
        return which_frames

    def select_volumes(self, condition, allow_one_off=False):
        """
        Selects the volumes that correspond to a specified condition.
        To actually load the selected volumes, use volume_manager.load_volumes()

        :param condition: the condition for which to select the volumes
        :type condition: Condition

        :param allow_one_off: whether to allow one slice in a volume to correspond to a different condition.
        Useful if the condition was too short or poorly aligned with the way you were capturing the volumes.
        :type allow_one_off: bool

        :return: 1D array of indices: a list of volumes corresponding to the condition.
        :rtype: numpy.ndarray
        """

        frames_for_condition = np.where(self.frame_to_condition == condition)
        vol_ids = self.volume_manager.frame_to_vol[frames_for_condition]
        volumes, counts = np.unique(vol_ids, return_counts=True)

        if allow_one_off:
            volumes = volumes[counts >= (self.volume_manager.fpv - 1)]
        else:
            volumes = volumes[counts == self.volume_manager.fpv]
        return volumes

    def __str__(self):
        description = ""
        description = description + f"Total frames : {self.n_frames}\n"
        description = description + f"Total cycles (ceil): {self.full_cycles}\n"
        description = description + self.cycle.__str__()
        return description

    def __repr__(self):
        return self.__str__()

    def summary(self):
        """
        Prints a detailed description of the experiment.
        """
        print(self.frame_manager.file_manager)
        print(self.cycle)
        print(f"Total cycles (ceil): {self.full_cycles}")
        print(self.volume_manager)

    @classmethod
    def from_spec(cls, spec):
        """
        Create experiment object from the specification dictionary.
        Necessary fields:

        __specify Cycle__
        'conditions' : list of conditions
        'timing' : timing of the conditions in the list

        __specify FileManager__
        'project_dir' :

        __specify VolumeManager__
        'frames_per_volume' :
        'first_good_frame' :

        :params spec: dictionary with the experiment specification
        :type spec: dict of str: many things ...
        """

        cycle = Cycle(spec['conditions'], spec['timing'])
        frame_manager = FrameManager(FileManager(spec['project_dir']))

        if 'first_good_frame' in spec:
            volume_manager = VolumeManager(spec['frames_per_volume'], frame_manager, fgf=spec['first_good_frame'])
        else:
            volume_manager = VolumeManager(spec['frames_per_volume'], frame_manager)

        return cls(frame_manager, volume_manager, cycle)

"""
This module contains the 'Experiment' class, which provides a summary of the information about an experiment. The
class can initialise, save, and load the database, search for frames based on volumes or annotations, and load image
data using the appropriate loader. To initialise the database, it integrates the information from the FileManager,
FrameManager, VolumeManager, as well as Annotations, to create a database.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any
import warnings

from numpy import ndarray

from .core import VolumeManager, ImageLoader
from .annotation import Annotation
from .dbmethods import DbReader, DbWriter


class Experiment:
    """
    The class can initialise, save, and load the database, search for frames based on volumes or annotations, and load image
    data using the appropriate loader. To initialise the database, it integrates the information from the File, Frame,
    and Volume managers, as well as Annotations, to create a database.

    Args:
        db_reader: a DbReader object connected to the database with the experiment description.

    Attributes:
        db: a DbReader object connected to the database with the experiment description.
        loader: an ImageLoader object to load metadata and image data from files.
    """

    def __init__(self, db_reader: DbReader):
        """
        Initialize the experiment with the given DbReader object.
        """

        assert isinstance(db_reader, DbReader), "Need DbReader to initialise the Experiment"

        self.db = db_reader
        # will add the loader the first time you are loading anything
        # in load_frames() or load_volumes()
        self.loader: ImageLoader

    @property
    def n_frames(self) -> int:
        """
        Returns the total number of frames in the experiment.
        """
        # TODO: cash this value when property is called for the first time
        return self.db.get_n_frames()

    @property
    def n_volumes(self) -> int:
        """
        Returns the total number of volumes in the experiment.
        This might include partial volumes at the beginning of the experiment (ID: -1)
        and at the end of the experiment (ID : -2).
        """
        # TODO: cash this value when property is called for the first time
        return len(self.db.get_volume_list())

    @property
    def n_full_volumes(self) -> int:
        """
        Returns the total number of full volumes in the experiment.
        """
        # TODO: cash this value when property is called for the first time
        options = self.db.get_options()
        return int(options['num_full_volumes'])

    @property
    def n_head_frames(self) -> int:
        """
        Returns the number of frames in the first partial volume,
        or 0 if there is no partial volume at the beginning.
        """
        # TODO: cash this value when property is called for the first time
        options = self.db.get_options()
        return int(options['num_head_frames'])

    @property
    def n_tail_frames(self) -> int:
        """
        Returns the number of frames in the last partial volume,
        or 0 if there is no partial volume at the end.
        """
        # TODO: cash this value when property is called for the first time
        options = self.db.get_options()
        return int(options['num_tail_frames'])

    @property
    def volumes(self) -> npt.NDArray:
        """
        Returns the list of volume IDs in the experiment.
        This might include partial volumes at the beginning of the experiment (ID: -1)
        and at the end of the experiment (ID : -2).
        """
        # TODO: cash this value when property is called for the first time
        volume_list = np.array(self.db.get_volume_list())
        if np.sum(volume_list == -1) > 0:
            warnings.warn(f"The are some frames at the beginning of the recording "
                          f"that don't correspond to a full volume.")
        if np.sum(volume_list == -2) > 0:
            warnings.warn(f"The are some frames at the end of the recording "
                          f"that don't correspond to a full volume.")
        return volume_list

    @property
    def annotations(self) -> List[str]:
        """
        Returns the list of annotation names that have been added to the experiment.
        """
        return self.db.get_Names_from_AnnotationTypes()

    @property
    def labels(self) -> dict:
        """
        Returns a dict with annotation names, labels and label descriptions
        that have been added to the experiment.
        """
        annotation_names = self.annotations
        label_dict = {}
        for annotation_name in annotation_names:
            label_names, descriptions = self.db.get_Name_and_Description_from_AnnotationTypeLabels(annotation_name)
            label_dict[annotation_name] = {
                'labels': label_names,
                'descriptions': descriptions}
        return label_dict

    @property
    def labels_df(self) -> pd.DataFrame:
        """
        Returns a dataframe with annotation names, labels and label descriptions
        that have been added to the experiment.
        """
        annotation_names = self.annotations
        label_dict = {'annotation': [], 'label': [], 'description': []}
        for annotation_name in annotation_names:
            label_names, descriptions = self.db.get_Name_and_Description_from_AnnotationTypeLabels(annotation_name)
            label_dict['annotation'].extend([annotation_name] * len(label_names))
            label_dict['label'].extend(label_names)
            label_dict['description'].extend([descriptions[key] for key in label_names])
        return pd.DataFrame(label_dict)

    @property
    def cycles(self) -> List[str]:
        """
        Returns the list of cycle names that have been added to the experiment.
        """
        return self.db.get_cycle_names()

    @property
    def file_names(self) -> List[str]:
        """
        Returns the list of file names that have been added to the experiment.
        """
        # TODO: cash this value when property is called for the first time
        return self.db.get_file_names()

    @property
    def frames_per_file(self) -> List[int]:
        """
        Returns the list of frames per file that have been added to the experiment.
        """
        # TODO: cash this value when property is called for the first time
        return self.db.get_frames_per_file()

    @property
    def data_dir(self) -> str:
        """
        Returns the path to the data directory.
        """
        # TODO: cash this value when property is called for the first time
        return self.db.get_data_dir()

    @property
    def frames_per_volume(self) -> int:
        """
        Returns the number of frames per volume.
        """
        # TODO: cash this value when property is called for the first time
        return self.db.get_fpv()

    @property
    def starting_slice(self) -> int:
        """
        Returns the number of the first slice in the experiment.
        """
        # TODO: cash this value when property is called for the first time
        return self.db.get_fgf() # fgf stands for first good frame


    @classmethod
    def create(cls, volume_manager: VolumeManager, annotations: List[Annotation], verbose: bool = False):
        """
        Creates a database instance from the core classes and initialises the experiment.

        Args:
            volume_manager: VolumeManager object that summarises the information about the image data.
            annotations: list of annotations to add to the experiment descriptions.
            verbose: whether to print the information about Filemanager, VolumeManager and Annotations on the screen.

        Returns:
            (Experiment): initialised experiment.
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

    @classmethod
    def from_dir(cls, dir_name: Union[Path, str], frames_per_volume: int,
                 starting_slice: int = 0, verbose: bool = False):
        """
        Creates a database instance from a directory and initialises the experiment.
        The directory should contain the image files.
        Annotations are not initialised, but can be added later.
        """
        # initialise volume manager
        volume_manager = VolumeManager.from_dir(dir_name, frames_per_volume, fgf=starting_slice)
        return cls.create(volume_manager, [], verbose)

    def save(self, file_name: Union[Path, str]):
        """
        Saves a database into a file.

        Args:
            file_name: full path to a file to save database.
                (Usually the filename would end with .db)
        """
        DbWriter(self.db.connection).save(file_name)

    def add_annotations(self, annotations: List[Annotation]):
        """
        Adds annotations to existing experiment.
        Does NOT save the changes to disc! run self.save() to save.

        Args:
            annotations: a list of annotations to add to the database.
        """
        DbWriter(self.db.connection).add_annotations(annotations)

    def add_annotations_from_df(self, annotation_df: pd.DataFrame,
                                cycles: Union[List[str], bool] = False, groups: Optional[str] = None,
                                info: Optional[dict] = None):
        """
        Adds annotations to existing experiment from a data frame.
        Does NOT save the changes to disc! run self.save() to save.

        Args:
            annotation_df: a dataframe with the annotation information
            cycles: a list of the annotation names that are cycles or a boolean.
                If False, all annotations are assumed to be timelines.
                If True, all annotations are assumed to be cycles.
            Specified as {'cycles': }
            groups: the group of the annotation if there are multiple groups in the dataframe.
                If None, all groups are added.
            info: additional information about the annotation, dictionary with keys:
                'annotation name': information
        """
        n_frames = self.db.get_n_frames()

        if groups is None:
            groups = annotation_df['group'].unique()

        annotations = []
        for group in groups:
            group_df = annotation_df[annotation_df['group'] == group]
            if cycles is True or (isinstance(cycles, list) and group in cycles):
                is_cycle = True
            else:
                is_cycle = False
            annotations.append(Annotation.from_df(n_frames, group_df, is_cycle, info))

        self.add_annotations(annotations)

    def delete_annotations(self, annotation_names: List[str]):
        """
        Deletes annotations from existing experiment.
        Does NOT save the changes to disc! run self.save() to save.

        Args:
            annotation_names: a list of annotation names to delete from the database.
        """
        for name in annotation_names:
            DbWriter(self.db.connection).delete_annotation(name)

    def close(self):
        """
        Close database connection.
        """
        self.db.connection.close()

    @classmethod
    def load(cls, file_name: Union[Path, str]):
        """
        Loads a database from a file and initialises an Experiment.

        Args:
            file_name: full path to a file to database.
        Return:
            (Experiment): initialised experiment.
        """
        db_reader = DbReader.load(file_name)
        return cls(db_reader)

    def choose_frames(self, conditions: Union[tuple, List[Tuple[str, str]]], logic: str = "and") -> List[int]:
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

    def choose_volumes(self, conditions: Union[tuple, List[Tuple[str, str]]], logic: str = "and",
                       verbose: bool = False) -> List[int]:
        """
        Selects only full volumes that correspond to specified conditions;
        Uses "or" or "and" between the conditions depending on logic.
        To load the selected volumes, use load_volumes()

        Args:
            verbose: Whether to print the information about how many frames were choose/ dropped
            conditions: a list of conditions on the annotation labels
                in a form [(group, name),(group, name), ...] where group is a string for the annotation type
                and name is the name of the label of that annotation type.
                For example [('light', 'on'), ('shape','c')]
            logic: "and" or "or" , default is "and".
        Returns:
            list of volumes that were chosen.
            Remember that frame numbers start at 1, but volumes start at 0.
        """
        # TODO : make all indices start at 1 ?

        assert isinstance(conditions, list) or isinstance(conditions, tuple), f"conditions must be a list or a tuple," \
                                                                              f" but got {type(conditions)} instead"
        if isinstance(conditions, tuple):
            conditions = [conditions]

        # get all the frames that correspond to the conditions
        frames = self.choose_frames(conditions, logic=logic)
        n_frames = len(frames)
        # leave only such frames that correspond to full volumes
        # TODO : not necessary to return the frames?
        volumes, frames = self.db.choose_full_volumes(frames)
        n_dropped = n_frames - len(frames)
        if verbose:
            print(f"Choosing only full volumes. "
                  f"Dropped {n_dropped} frames, kept {len(frames)}")

        return volumes

    def load_volumes(self, volumes: Union[npt.NDArray, List[int]], verbose: bool = False) -> npt.NDArray:
        """
        Load volumes. Will load the specified full volumes.
        All the returned volumes or slices should have the same number of frames in them.

        Args:
            volumes: the indexes of volumes to load.
            verbose: Whether to print the information about the loading
        Returns:
            4D array with the loaded volumes. TZYX order.
        """
        # if array convert to list of int
        if isinstance(volumes, np.ndarray):
            # make sure it is a 1D array
            assert len(volumes.shape) == 1, "volumes must be a 1D array"

            # make sure all the volumes can be safely converted to integers
            assert np.all(volumes.astype(int) == volumes), "All the volumes must be integers"
            volumes = volumes.astype(int).tolist()

        frames = self.db.get_frames_per_volumes(volumes)
        info = self.db.prepare_frames_for_loading(frames)

        # unpack
        data_dir, file_names, file_ids, frame_in_file, volumes_per_frame = info
        # get unique volumes and check that they are the same as the ones we asked for
        assert set(volumes_per_frame) == set(volumes), "Requested volumes" \
                                                       f" {set(volumes).difference(set(volumes_per_frame))} " \
                                                       "can not be found"
        # make full paths to files ( remember file ids start with 1 )
        files = [Path(data_dir, file_names[file_id - 1]) for file_id in file_ids]
        if not hasattr(self, "loader"):
            self.loader = ImageLoader(Path(data_dir, file_names[0]))
        volumes_img = self.loader.load_volumes(frame_in_file,
                                               files,
                                               volumes_per_frame,
                                               show_file_names=False,
                                               show_progress=verbose)
        return volumes_img

    def get_volume_annotations(self, volumes: Union[npt.NDArray, List[int]],
                               annotation_names: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Get annotations for volumes.
        Will get the labels for the specified full volumes from each available annotation.

        Args:
            volumes: the indexes of volumes to get annotation for. If a multidimensional array is passed,
                will flatten it and get annotations for all the volumes in it.
            annotation_names: the names of the annotations to get. If None, will get all the annotations.

        Returns:
            a dictionary with the annotations for each annotation type.
            The keys are the annotation types, the values are lists of labels for each volume.
            The last key is "volumes" and the value is a list of volumes.
        """
        # TODO: throw a warning if some volumes are not in the database

        # if array convert to list of int
        if isinstance(volumes, np.ndarray):
            # turn into a 1D array
            volumes = volumes.flatten()
            # make sure all the volumes can be safely converted to integers
            assert np.all(volumes.astype(int) == volumes), "All the volumes must be integers"
            volumes = volumes.astype(int).tolist()

        print(f'volumes : {volumes}')
        # get annotations for the volumes
        annotations = self.db.get_volume_annotations(volumes, annotation_names=annotation_names)

        # prepare dict for the annotations
        annotation = {key: [] for key in annotations.keys()}
        annotation["volumes"] = []

        # get a single label per volume
        for volume in volumes:
            for group, data in annotations.items():
                volume_ids = np.array(data["volume_ids"])
                labels = np.array(data["labels"])
                # check that the volume has the same labels
                labels_per_volume = set(labels[volume_ids == volume])
                if len(labels_per_volume) > 1:
                    raise ValueError(f"Volume {volume} has different labels ({labels_per_volume}) "
                                     f"for the same annotation {group}. Can't assign a single label to the volume.")
                # add the label to the dict
                annotation[group].append(list(labels_per_volume)[0])
            # add the volume to the dict
            annotation["volumes"].append(volume)

        return annotation

    def load_slices(self, slices: List[int], volumes: List[int] = None,
                    skip_missing: bool = False, verbose: bool = False) -> npt.NDArray:
        """
        Load volumes. Will load the specified full volumes.
        All the returned volumes or slices should have the same number of frames in them.

        Args:
            slices: the indexes of slices in the volumes to load.
            volumes: the indexes of volumes to load slices for. If None, will load slices for all volumes.
            skip_missing: Whether to skip missing volumes.
                If False, will raise an error if a slice is missing for any volume.
            verbose: Whether to print the information about the loading
        Returns:
            4D array with the loaded slices for selected volumes. TZYX order.
        """
        if volumes is None:
            volumes = self.db.get_volume_list()

        frames = self.db.get_frames_per_volumes(volumes, slices=slices)
        info = self.db.prepare_frames_for_loading(frames)

        # unpack and load
        data_dir, file_names, file_ids, frame_in_file, volumes_per_frame = info

        # get unique volumes and check that they are the same as the ones we asked for
        if skip_missing:  # throw a warning
            if set(volumes_per_frame) != set(volumes):
                warnings.warn(f"Requested volumes {set(volumes).difference(set(volumes_per_frame))} " +
                              f"are not present in the slices {slices}. " +
                              f"Loaded slices for {set(volumes_per_frame)} volumes.")
        else:  # throw an error
            assert set(volumes_per_frame) == set(volumes), \
                f"Requested volumes {set(volumes).difference(set(volumes_per_frame))} " \
                f"are not present in the slices {slices}. "

        # make full paths to files ( remember file ids start with 1 )
        files = [Path(data_dir, file_names[file_id - 1]) for file_id in file_ids]
        if not hasattr(self, "loader"):
            self.loader = ImageLoader(Path(data_dir, file_names[0]))

        volumes_img = self.loader.load_volumes(frame_in_file,
                                               files,
                                               volumes_per_frame,
                                               show_file_names=False,
                                               show_progress=verbose)

        # if the z dimension is smaller than the number of slices, throw a warning
        if volumes_img.shape[1] < len(slices):
            warnings.warn(f"Some of the requested slices {slices} are not present in the volumes. " +
                          f"Loaded {volumes_img.shape[1]} slices instead of {len(slices)}")

        return volumes_img

    def list_volumes(self) -> npt.NDArray[int]:
        """
        Returns a list of all the volumes IDs in the experiment.
        If partial volumes are present: for "head" returns -1, for "tail" returns -2.

        Returns:
            list of volume IDs
        """
        #TODO : Remove this function and use volumes property instead
        warnings.warn(f"list_volumes will be removed in vodex 1.1.0 use volumes property instead.")

        return self.volumes

    def list_conditions_per_cycle(self, annotation_type: str, as_volumes: bool = True) -> Tuple[List[int], List[str]]:
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
        names = self.db._get_Names_from_AnnotationTypeLabels()

        return condition_ids, names

    def list_cycle_iterations(self, annotation_type: str, as_volumes: bool = True) -> List[int]:
        """
        Returns a list of cycle iterations for a specified annotation.
        The annotation must have been initialised from a cycle.

        Args:
            annotation_type: The name of the annotation for which to get the cycle iteratoins list
            as_volumes: weather to return cycle iteratoins per frame ( default) or per volume.
                If as_volumes is true, it is expected that the cycle iteratoins are not changing in the middle of the volume.
                Will throw an error if it happens.
            as_volumes: bool
        Returns:
            list of the condition ids (cycle iterations per frame or per volume)
        """

        if as_volumes:
            _, cycle_its, count = self.db.get_cycleIterations_per_volumes(annotation_type)
            fpv = self.db.get_fpv()
            assert np.all(np.array(count) == fpv), "Can't list_cycle_iterations with as_volumes=True: " \
                                                   "some iterations don't cover the whole volume." \
                                                   "You might want to get iterations per frame," \
                                                   " by setting as_volumes=False"
        else:
            _, cycle_its = self.db.get_cycleIterations_per_frame(annotation_type)

        return cycle_its

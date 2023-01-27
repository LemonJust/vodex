"""
This module contains the 'Experiment' class, which provides a summary of the information about an experiment. The
class can initialise, save, and load the database, search for frames based on volumes or annotations, and load image
data using the appropriate loader. To initialise the database, it integrates the information from the FileManager,
FrameManager, VolumeManager, as well as Annotations, to create a database.
"""

import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Union, List, Tuple
import warnings

from .core import VolumeManager, Annotation, ImageLoader
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
                and name is the name of the label of that annotation type. For example [('light', 'on'), ('shape','c')]
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

    def load_volumes(self, volumes: List[int], verbose: bool = False) -> npt.NDArray:
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
        if not hasattr(self, "loader"):
            self.loader = ImageLoader(Path(data_dir, file_names[0]))
        volumes_img = self.loader.load_volumes(frame_in_file,
                                               files,
                                               volumes,
                                               show_file_names=False,
                                               show_progress=verbose)
        return volumes_img

    def list_volumes(self) -> npt.NDArray[int]:
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


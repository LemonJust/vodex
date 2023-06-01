"""
Tests for the `vodex.experiment` module.
"""
import pytest
import sqlite3
from vodex import *
from pathlib import Path

import numpy as np
import pandas as pd

from .conftest import TEST_DATA, SPLIT_MOVIE_DIR, \
    VOLUMES_0_1, HALF_VOLUMES_0_1, VOLUMES_0_TAIL_SLICES_0_1, SLICES_0_1, SLICES_0, SLICES_2, \
    VOLUMES_TAIL, VOLUME_M, \
    SHAPE_AN, CNUM_AN, LIGHT_AN, \
    SHAPE_CYCLE, CNUM_CYCLE, LIGHT_TML


# Fixture to create an Experiment instance in agreement with the test database
@pytest.fixture
def experiment():
    # Create the experiment and return it
    return Experiment.create(VOLUME_M, [SHAPE_AN, CNUM_AN, LIGHT_AN])


# Fixture to create an Experiment without annotations
@pytest.fixture
def experiment_no_annotations():
    # data to create an experiment
    data_dir_split = Path(TEST_DATA, "test_movie")

    volume_m = VolumeManager.from_dir(data_dir_split, 10, fgf=0)
    # Create the experiment and return it
    return Experiment.create(volume_m, [])


def test_create(experiment):
    assert isinstance(experiment, Experiment)
    assert isinstance(experiment.db, DbReader)
    with pytest.raises(AttributeError):
        experiment.loader  # the loader is not initialized yet


def test_create_verbose():
    experiment = Experiment.create(VOLUME_M, [SHAPE_AN, CNUM_AN, LIGHT_AN], verbose=True)
    assert isinstance(experiment, Experiment)
    assert isinstance(experiment.db, DbReader)
    with pytest.raises(AttributeError):
        experiment.loader  # the loader is not initialized yet


def test_from_dir():
    data_dir_split = Path(TEST_DATA, "test_movie")
    experiment = Experiment.from_dir(data_dir_split, 10, starting_slice=0)
    assert isinstance(experiment, Experiment)
    assert isinstance(experiment.db, DbReader)
    with pytest.raises(AttributeError):
        experiment.loader  # the loader is not initialized yet


def test_save(experiment):
    # Create the test db file
    test_db_file = 'test_save.db'
    experiment.save(test_db_file)
    assert Path(test_db_file).is_file()
    # Clean up: delete the test file
    Path(test_db_file).unlink()


def test_n_frames_property(experiment, experiment_no_annotations):
    assert experiment.n_frames == 42
    assert experiment_no_annotations.n_frames == 42


def test_n_volumes_property(experiment, experiment_no_annotations):
    assert experiment.n_volumes == 5
    assert experiment_no_annotations.n_volumes == 5


def test_n_full_volumes_property(experiment, experiment_no_annotations):
    assert experiment.n_full_volumes == 4
    assert experiment_no_annotations.n_full_volumes == 4


def test_n_head_frames_property(experiment, experiment_no_annotations):
    assert experiment.n_head_frames == 0
    assert experiment_no_annotations.n_head_frames == 0


def test_n_tail_frames_property(experiment, experiment_no_annotations):
    assert experiment.n_tail_frames == 2
    assert experiment_no_annotations.n_tail_frames == 2


def test_volumes_property(experiment, experiment_no_annotations):
    assert (experiment.volumes == [-2, 0, 1, 2, 3]).all()
    assert (experiment_no_annotations.volumes == [-2, 0, 1, 2, 3]).all()


def test_annotations_property(experiment, experiment_no_annotations):
    assert experiment.annotations == ['shape', 'c label', 'light']
    assert experiment_no_annotations.annotations == []


def test_labels_property(experiment, experiment_no_annotations):
    assert experiment.labels == {'c label': {
        'descriptions': {'c1': 'written c1',
                         'c2': 'written c2',
                         'c3': None},
        'labels': ['c1', 'c2', 'c3']},
        'light': {
            'descriptions': {'off': 'the intensity of the background is low',
                             'on': 'the intensity of the background is high'},
            'labels': ['off', 'on']},
        'shape': {
            'descriptions': {'c': 'circle on the screen',
                             's': 'square on the screen'},
            'labels': ['c', 's']}}
    assert experiment_no_annotations.labels == {}


def test_labels_df_property(experiment, experiment_no_annotations):
    assert experiment.labels_df.equals(pd.DataFrame({'annotation':
                                                         ['shape', 'shape',
                                                          'c label', 'c label', 'c label',
                                                          'light', 'light'],
                                                     'label': ['c', 's',
                                                               'c1', 'c2', 'c3',
                                                               'off', 'on'],
                                                     'description': ['circle on the screen',
                                                                     'square on the screen',
                                                                     'written c1', 'written c2', None,
                                                                     'the intensity of the background is low',
                                                                     'the intensity of the background is high']}))
    assert experiment_no_annotations.labels_df.empty


def test_cycles_property(experiment, experiment_no_annotations):
    assert experiment.cycles == ['c label', 'shape']
    assert experiment_no_annotations.cycles == []


def test_file_names_property(experiment, experiment_no_annotations):
    assert experiment.file_names == ['mov0.tif', 'mov1.tif', 'mov2.tif']
    assert experiment_no_annotations.file_names == ['mov0.tif', 'mov1.tif', 'mov2.tif']


def test_frames_per_file_property(experiment, experiment_no_annotations):
    assert experiment.frames_per_file == [7, 18, 17]
    assert experiment_no_annotations.frames_per_file == [7, 18, 17]


def test_data_dir_property(experiment, experiment_no_annotations):
    assert experiment.data_dir == SPLIT_MOVIE_DIR.as_posix()
    assert experiment_no_annotations.data_dir == SPLIT_MOVIE_DIR.as_posix()


def test_frames_per_volume_property(experiment, experiment_no_annotations):
    assert experiment.frames_per_volume == 10
    assert experiment_no_annotations.frames_per_volume == 10


def test_starting_slice_property(experiment, experiment_no_annotations):
    assert experiment.starting_slice == 0
    assert experiment_no_annotations.starting_slice == 0


def test_add_annotations(experiment_no_annotations):
    experiment_no_annotations.add_annotations([SHAPE_AN])

    # Test that the annotations have been added
    # (if it was added successfully, there should be exactly 20 such rows)
    cursor = experiment_no_annotations.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 2;")
    labels = [row[0] for row in cursor.fetchall()]
    assert labels == [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]


def test_add_annotations_from_df(experiment_no_annotations):
    shape_df = SHAPE_CYCLE.to_df()
    cnum_df = CNUM_CYCLE.to_df()
    light_df = LIGHT_TML.to_df()

    # add annotations
    experiment_no_annotations.add_annotations_from_df(shape_df, cycles=True)
    # (if it was added successfully, there should be exactly 20 such rows)
    cursor = experiment_no_annotations.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 2;")
    labels = [row[0] for row in cursor.fetchall()]
    assert labels == [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

    two_groups_df = pd.concat([cnum_df, light_df])
    experiment_no_annotations.add_annotations_from_df(two_groups_df, cycles=['c label'])
    cursor = experiment_no_annotations.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 7;")
    labels = [row[0] for row in cursor.fetchall()]
    assert labels == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    cursor = experiment_no_annotations.db.connection.execute(
        "SELECT Name FROM AnnotationTypes")
    names = [row[0] for row in cursor.fetchall()]
    assert names == ["c label", "light", "shape"]

    cursor = experiment_no_annotations.db.connection.execute(
        "SELECT Name, Description FROM AnnotationTypeLabels")
    info = cursor.fetchall()
    names = [row[0] for row in info]
    description = [row[1] for row in info]
    assert names == ["c", "s", "c1", "c2", "c3", "off", "on"]
    assert description == ["circle on the screen", "square on the screen",
                           "written c1", "written c2", None,
                           "the intensity of the background is low",
                           "the intensity of the background is high"]


def test_delete_annotations(experiment):
    cursor = experiment.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 1;")
    assert len(cursor.fetchall()) == 22
    cursor = experiment.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 2;")
    assert len(cursor.fetchall()) == 20

    cursor = experiment.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 6;")
    assert len(cursor.fetchall()) == 20

    cursor = experiment.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 3;")
    assert len(cursor.fetchall()) == 20
    cursor = experiment.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 4;")
    assert len(cursor.fetchall()) == 12

    experiment.delete_annotations(["shape", "light"])

    # Test that the annotations have been deleted
    cursor = experiment.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 1;")
    assert len(cursor.fetchall()) == 0
    cursor = experiment.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 2;")
    assert len(cursor.fetchall()) == 0

    cursor = experiment.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 6;")
    assert len(cursor.fetchall()) == 0

    # c-label annotation should stay untouched:
    # (if it was added successfully, there should be exactly 20 such rows)
    cursor = experiment.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 3;")
    assert len(cursor.fetchall()) == 20
    cursor = experiment.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 4;")
    assert len(cursor.fetchall()) == 12


def test_close(experiment_no_annotations):
    experiment_no_annotations.close()
    # Test that the connection has been closed
    with pytest.raises(sqlite3.ProgrammingError):
        experiment_no_annotations.db.connection.execute("SELECT * FROM Options LIMIT 1;")


def test_load(TEST_DB):
    experiment = Experiment.load(TEST_DB)
    assert isinstance(experiment, Experiment)
    assert isinstance(experiment.db, DbReader)
    with pytest.raises(AttributeError):
        experiment.loader  # the loader is not initialized yet


def test_choose_frames(experiment):
    conditions1 = [("light", "on"), ("light", "off")]
    conditions2 = [("light", "on")]
    conditions3 = [("light", "on"), ("c label", "c1")]
    conditions4 = [("light", "on"), ("c label", "c2")]
    conditions5 = [("light", "on"), ("c label", "c2"), ("c label", "c3")]
    conditions6 = [("light", "on"), ("c label", "c2"), ("shape", "s")]

    # correct answers
    frames_and1 = []
    frames_and2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    frames_and3 = []
    frames_and4 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    frames_and5 = []
    frames_and6 = [11, 12, 13, 14, 15]

    frames_or1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                  31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                  41, 42]
    frames_or2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    frames_or3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                  31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    frames_or4 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                  41, 42]
    frames_or5 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                  41, 42]
    frames_or6 = [6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                  31, 32, 33, 34, 35,
                  41, 42]

    frames = experiment.choose_frames(conditions1, logic="and")
    assert frames_and1 == frames
    frames = experiment.choose_frames(conditions2, logic="and")
    assert frames_and2 == frames
    frames = experiment.choose_frames(conditions3, logic="and")
    assert frames_and3 == frames
    frames = experiment.choose_frames(conditions4, logic="and")
    assert frames_and4 == frames
    frames = experiment.choose_frames(conditions5, logic="and")
    assert frames_and5 == frames
    frames = experiment.choose_frames(conditions6, logic="and")
    assert frames_and6 == frames

    frames = experiment.choose_frames(conditions1, logic="or")
    assert frames_or1 == frames
    frames = experiment.choose_frames(conditions2, logic="or")
    assert frames_or2 == frames
    frames = experiment.choose_frames(conditions3, logic="or")
    assert frames_or3 == frames
    frames = experiment.choose_frames(conditions4, logic="or")
    assert frames_or4 == frames
    frames = experiment.choose_frames(conditions5, logic="or")
    assert frames_or5 == frames
    frames = experiment.choose_frames(conditions6, logic="or")
    assert frames_or6 == frames


def test_choose_volumes(experiment):
    conditions1 = [("light", "on"), ("light", "off")]
    conditions2 = [("light", "on")]
    conditions3 = [("light", "on"), ("c label", "c1")]
    conditions4 = [("light", "on"), ("c label", "c2")]
    conditions5 = [("light", "on"), ("c label", "c2"), ("c label", "c3")]
    conditions6 = [("light", "on"), ("c label", "c2"), ("shape", "s")]

    # correct answers
    volumes_and1 = []
    volumes_and2 = [1, 2]
    volumes_and3 = []
    volumes_and4 = [1]
    volumes_and5 = []
    volumes_and6 = []

    volumes_or1 = [0, 1, 2, 3]
    volumes_or2 = [1, 2]
    volumes_or3 = [0, 1, 2, 3]
    volumes_or4 = [1, 2]
    volumes_or5 = [1, 2]
    volumes_or6 = [1, 2]

    frames = experiment.choose_volumes(conditions1, logic="and")
    assert volumes_and1 == frames
    frames = experiment.choose_volumes(conditions2, logic="and")
    assert volumes_and2 == frames
    frames = experiment.choose_volumes(conditions3, logic="and")
    assert volumes_and3 == frames
    frames = experiment.choose_volumes(conditions4, logic="and")
    assert volumes_and4 == frames
    frames = experiment.choose_volumes(conditions5, logic="and")
    assert volumes_and5 == frames
    frames = experiment.choose_volumes(conditions6, logic="and")
    assert volumes_and6 == frames

    frames = experiment.choose_volumes(conditions1, logic="or")
    assert volumes_or1 == frames
    frames = experiment.choose_volumes(conditions2, logic="or")
    assert volumes_or2 == frames
    frames = experiment.choose_volumes(conditions3, logic="or")
    assert volumes_or3 == frames
    frames = experiment.choose_volumes(conditions4, logic="or")
    assert volumes_or4 == frames
    frames = experiment.choose_volumes(conditions5, logic="or")
    assert volumes_or5 == frames
    frames = experiment.choose_volumes(conditions6, logic="or")
    assert volumes_or6 == frames


def test_load_volumes(experiment_no_annotations):
    volumes_img = experiment_no_annotations.load_volumes([0, 1])
    assert (VOLUMES_0_1 == volumes_img).all()

    volumes_img = experiment_no_annotations.load_volumes([-2])
    assert (VOLUMES_TAIL == volumes_img).all()

    with pytest.raises(AssertionError):
        experiment_no_annotations.load_volumes([1, -2])

    volumes_img = experiment_no_annotations.load_volumes(np.array([0, 1]))
    assert (VOLUMES_0_1 == volumes_img).all()

    volumes_img = experiment_no_annotations.load_volumes(np.array([-2]))
    assert (VOLUMES_TAIL == volumes_img).all()

    with pytest.raises(AssertionError) as e:
        experiment_no_annotations.load_volumes(np.array([[0, 1], [0, 1]]))
    assert "volumes must be a 1D array" in str(e.value)

    with pytest.raises(AssertionError) as e:
        experiment_no_annotations.load_volumes(np.array([0, 1, 2.3]))
    assert "All the volumes must be integers" in str(e.value)


def test_get_volume_annotations(experiment):
    with pytest.raises(ValueError) as e:
        experiment.get_volume_annotations([0])
    assert "Can't assign a single label to the volume." in str(e.value)

    volume_annotations = experiment.get_volume_annotations([0], annotation_names=["light", "c label"])
    assert volume_annotations == {'light': ['off'], 'c label': ['c1'], 'volumes': [0]}

    annotation = {'light': ['off', 'off', 'on'], 'c label': ['c2', 'c1', 'c2'], 'volumes': [-2, 0, 1]}
    volume_annotations = experiment.get_volume_annotations([-2, 0, 1], annotation_names=["light", "c label"])
    assert volume_annotations == annotation

    volume_annotations = experiment.get_volume_annotations(np.array([-2, 0, 1]), annotation_names=["light", "c label"])
    assert volume_annotations == annotation

    volume_annotations = experiment.get_volume_annotations(np.array([-2.0, 0.0, 1.0]),
                                                           annotation_names=["light", "c label"])
    assert volume_annotations == annotation

    # 2D array
    volume_annotations = experiment.get_volume_annotations(np.array([[-2], [0], [1]]),
                                                           annotation_names=["light", "c label"])
    assert volume_annotations == annotation

    with pytest.raises(AssertionError) as e:
        experiment.get_volume_annotations(np.array([1.2, 3]))
    assert "All the volumes must be integers" in str(e.value)


def test_load_slices(experiment_no_annotations):
    volumes_img = experiment_no_annotations.load_slices([0, 1, 2, 3, 4], volumes=[0, 1])
    assert (HALF_VOLUMES_0_1 == volumes_img).all()

    volumes_img = experiment_no_annotations.load_slices([0, 1], volumes=[-2])
    assert (VOLUMES_TAIL == volumes_img).all()

    volumes_img = experiment_no_annotations.load_slices([0, 1], volumes=[0, -2])
    assert (VOLUMES_0_TAIL_SLICES_0_1 == volumes_img).all()

    volumes_img = experiment_no_annotations.load_slices([0])
    assert (SLICES_0 == volumes_img).all()

    volumes_img = experiment_no_annotations.load_slices([0, 1])
    assert (SLICES_0_1 == volumes_img).all()

    with pytest.raises(AssertionError) as e:
        volumes_img = experiment_no_annotations.load_slices([2])
    assert str(e.value) == "Requested volumes {-2} are not present in the slices [2]. "

    with pytest.warns(UserWarning) as record:
        volumes_img = experiment_no_annotations.load_slices([2], skip_missing=True)
        assert (SLICES_2 == volumes_img).all()
    assert str(record[0].message) == \
           "Requested volumes {-2} are not present in the slices [2]. Loaded slices for {0, 1, 2, 3} volumes."

    with pytest.raises(AssertionError) as e:
        experiment_no_annotations.load_slices([1, 2], volumes=[1, -2])
    assert str(e.value) == "Can't have different number of frames per volume!"

    # test that the warning is called
    with pytest.warns(UserWarning) as record:
        volumes_img = experiment_no_annotations.load_slices([0, 15])
        assert (SLICES_0 == volumes_img).all()
    assert str(record[0].message) == \
           "Some of the requested slices [0, 15] are not present in the volumes. Loaded 1 slices instead of 2"


def test_list_volumes(experiment_no_annotations):
    volumes_list = experiment_no_annotations.list_volumes()
    assert (volumes_list == [-2, 0, 1, 2, 3]).all()


def test_list_conditions_per_cycle(experiment):
    with pytest.raises(AssertionError):
        experiment.list_conditions_per_cycle("shape", as_volumes=True)

    ids, names = experiment.list_conditions_per_cycle("shape", as_volumes=False)
    shape_id_per_cycle = [1, 1, 1, 1, 1,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          1, 1, 1, 1, 1]

    assert ids == shape_id_per_cycle
    assert names == ["c", "s", 'c1', 'c2', 'c3', "on", "off"]

    ids, names = experiment.list_conditions_per_cycle("c label", as_volumes=False)
    assert ids == [3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                   5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    assert names == ["c", "s", 'c1', 'c2', 'c3', "on", "off"]

    ids, names = experiment.list_conditions_per_cycle("c label", as_volumes=True)
    assert ids == [3, 4, 5]
    assert names == ["c", "s", 'c1', 'c2', 'c3', "on", "off"]


def test_list_cycle_iterations(experiment):
    with pytest.raises(AssertionError):
        experiment.list_cycle_iterations("shape", as_volumes=True)

    ids = experiment.list_cycle_iterations("shape", as_volumes=False)
    cycle_per_frame = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       2, 2]
    assert ids == cycle_per_frame

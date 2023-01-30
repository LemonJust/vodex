"""
Tests for the `vodex.experiment` module.
"""
import pytest
import sqlite3
import tifffile as tif
from vodex import *
from pathlib import Path

TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')
SPLIT_MOVIE_DIR = Path(TEST_DATA, "test_movie")
TEST_DB = Path(TEST_DATA, "test.db")
# saved images of volumes 0 and 1 and the last two frames (tail)
VOLUMES_0_1 = tif.imread(Path(TEST_DATA, 'loader_test', "volumes_1_2.tif"))
VOLUMES_TAIL = tif.imread(Path(TEST_DATA, 'loader_test', "volumes_tail.tif"))

VOLUME_M = VolumeManager.from_dir(SPLIT_MOVIE_DIR, 10, fgf=0)


# annotations to create an experiment
def prepare_annotations():
    shape = Labels("shape", ["c", "s"],
                   state_info={"c": "circle on the screen", "s": "square on the screen"})
    light = Labels("light", ["on", "off"], group_info="Information about the light",
                   state_info={"on": "the intensity of the background is high",
                               "off": "the intensity of the background is low"})
    cnum = Labels("c label", ['c1', 'c2', 'c3'], state_info={'c1': 'written c1', 'c2': 'written c1'})

    shape_cycle = Cycle([shape.c, shape.s, shape.c], [5, 10, 5])
    cnum_cycle = Cycle([cnum.c1, cnum.c2, cnum.c3], [10, 10, 10])
    light_tml = Timeline([light.off, light.on, light.off], [10, 20, 12])

    shape_an = Annotation.from_cycle(42, shape, shape_cycle)
    cnum_an = Annotation.from_cycle(42, cnum, cnum_cycle)
    light_an = Annotation.from_timeline(42, light, light_tml)

    return shape_an, cnum_an, light_an


SHAPE_AN, CNUM_AN, LIGHT_AN = prepare_annotations()


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


def test_save(experiment):
    # Create the test db file
    test_db_file = 'test_save.db'
    experiment.save(test_db_file)
    assert Path(test_db_file).is_file()
    # Clean up: delete the test file
    Path(test_db_file).unlink()


def test_add_annotations(experiment_no_annotations):
    experiment_no_annotations.add_annotations([SHAPE_AN])

    # Test that the annotations have been added
    # (if it was added successfully, there should be exactly 20 such rows)
    cursor = experiment_no_annotations.db.connection.execute(
        "SELECT FrameId FROM Annotations WHERE AnnotationTypeLabelId = 2;")
    labels = [row[0] for row in cursor.fetchall()]
    assert labels == [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]


def test_close(experiment_no_annotations):
    experiment_no_annotations.close()
    # Test that the connection has been closed
    with pytest.raises(sqlite3.ProgrammingError):
        experiment_no_annotations.db.connection.execute("SELECT * FROM Options LIMIT 1;")


def test_load():
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

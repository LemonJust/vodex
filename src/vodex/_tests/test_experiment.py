"""
Tests for the `vodex.experiment` module.
"""
import pytest
import tifffile as tif
from vodex import *
from pathlib import Path

TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')


class TestExperiment:
    # data to create an experiment
    data_dir_split = Path(TEST_DATA, "test_movie")

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
    annotations = [shape_an, cnum_an, light_an]

    volume_m = VolumeManager.from_dir(data_dir_split, 10, fgf=0)

    def test_create_and_save(self):
        experiment = Experiment.create(self.volume_m, self.annotations)
        experiment.save(Path(TEST_DATA, "test_experiment.db"))

    def test_load(self):
        # not sure how to compare really
        Experiment.load(Path(TEST_DATA, "test.db"))

    def test_choose_frames(self):
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

        experiment = Experiment.load(Path(TEST_DATA, "test.db"))

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

    def test_choose_volumes(self):
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

        experiment = Experiment.load(Path(TEST_DATA, "test.db"))

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

    def test_load_volumes(self):
        volumes1 = [0, 1]
        volumes2 = [-2]
        volumes3 = [1, -2]

        experiment = Experiment.load(Path(TEST_DATA, "test.db"))
        volumes_img = experiment.load_volumes(volumes1)
        volumes_0_1 = tif.imread(Path(TEST_DATA, 'loader_test', "volumes_1_2.tif"))
        assert (volumes_0_1 == volumes_img).all()

        volumes_img = experiment.load_volumes(volumes2)
        volumes_tail = tif.imread(Path(TEST_DATA, 'loader_test', "volumes_tail.tif"))
        assert (volumes_tail == volumes_img).all()

        with pytest.raises(AssertionError):
            experiment.load_volumes(volumes3)

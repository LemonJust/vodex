"""
Tests for the `vodex.core` module.
"""
import json
from pathlib import Path
import pytest
from vodex import *
import pandas as pd

from .conftest import TEST_DATA


class TestFrameManager:
    data_dir_split = Path(TEST_DATA, "test_movie")
    file_m = FileManager(data_dir_split)
    frame_to_file = [0, 0, 0, 0, 0, 0, 0,  # 7
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 18
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # 17
    frame_in_file = [0, 1, 2, 3, 4, 5, 6,  # 7
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,  # 18
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 17

    def test_eq(self):
        frame_m1 = FrameManager(self.file_m)
        frame_m2 = FrameManager(self.file_m)
        assert frame_m1 == frame_m2
        assert frame_m2 == frame_m1
        assert frame_m1.__eq__("FrameManager") == NotImplemented

    def test_get_frame_mapping(self):
        frame_m = FrameManager(self.file_m)
        frame_to_file, frame_in_file = frame_m._get_frame_mapping()

        assert frame_to_file == self.frame_to_file
        assert frame_in_file == self.frame_in_file

    def test_from_dir(self):
        frame_m1 = FrameManager(self.file_m)
        frame_m2 = FrameManager.from_dir(self.data_dir_split)
        assert frame_m1 == frame_m2

    def test_str(self):
        frame_m1 = FrameManager(self.file_m)
        assert str(frame_m1) == "Total 42 frames."

    def test_repr(self):
        frame_m1 = FrameManager(self.file_m)
        assert repr(frame_m1) == "Total 42 frames."


class TestVolumeManager:
    data_dir_split = Path(TEST_DATA, "test_movie")
    file_m = FileManager(data_dir_split)
    frame_m = FrameManager(file_m)
    # TODO : test with fgf not 0
    volume_m = VolumeManager(10, frame_m, fgf=0)

    frame_to_vol = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                    -2, -2]

    frame_to_z = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  0, 1]

    def test_get_frames_to_z_mapping(self):
        frame_to_z = self.volume_m._get_frames_to_z_mapping()
        assert frame_to_z == self.frame_to_z

    def test_get_frames_to_volumes_mapping(self):
        frame_to_vol = self.volume_m._get_frames_to_volumes_mapping()
        assert frame_to_vol == self.frame_to_vol

    def test_from_dir(self):
        volume_m = VolumeManager.from_dir(self.data_dir_split, 10, fgf=0)
        assert self.volume_m == volume_m

    def test_eq(self):
        assert self.volume_m.__eq__("VolumeManager") == NotImplemented

    def test_repr(self):
        assert repr(
            self.volume_m) == 'Total frames : 42\nVolumes start on frame : 0\nTotal good volumes : 4\nFrames per volume : 10\nTailing frames (not a full volume , at the end) : 2\n'


class TestAnnotation:
    shape = Labels("shape", ["c", "s"],
                   state_info={"c": "circle on the screen", "s": "square on the screen"})
    shape_cycle = Cycle([shape.c, shape.s, shape.c], [5, 10, 5])
    shape_timeline = Timeline([shape.c, shape.s, shape.c, shape.s, shape.c],
                              [5, 10, 10, 10, 7])

    shape_frame_to_label = [shape.c] * 5
    shape_frame_to_label.extend([shape.s] * 10)
    shape_frame_to_label.extend([shape.c] * 10)
    shape_frame_to_label.extend([shape.s] * 10)
    shape_frame_to_label.extend([shape.c] * 7)

    frame_to_cycle = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      2, 2]

    def test_get_timeline(self):
        a = Annotation.from_timeline(42, self.shape, self.shape_timeline)
        shape_timeline = a.get_timeline()
        assert self.shape_timeline == shape_timeline
        assert shape_timeline == self.shape_timeline

    def test_from_cycle(self):
        a1 = Annotation(42, self.shape, self.shape_frame_to_label)
        a2 = Annotation.from_cycle(42, self.shape, self.shape_cycle)

        assert a1.frame_to_label == a2.frame_to_label
        assert a1.n_frames == a2.n_frames
        assert a1.labels == a2.labels
        assert a1.name == a2.name

        assert a1.cycle is None
        assert a2.cycle == self.shape_cycle
        assert a2.frame_to_cycle == self.frame_to_cycle

    def test_from_timeline(self):
        a1 = Annotation(42, self.shape, self.shape_frame_to_label)
        a2 = Annotation.from_timeline(42, self.shape, self.shape_timeline)
        a3 = Annotation.from_cycle(42, self.shape, self.shape_cycle)

        assert a1 == a2
        assert a2 == a1

        assert a3 != a2
        assert a2 != a3

    def test_eq(self):
        a1 = Annotation.from_cycle(42, self.shape, self.shape_cycle)
        a2 = Annotation.from_cycle(42, self.shape, self.shape_cycle)

        assert a1 == a2
        assert a2 == a1
        assert a1.__eq__("Annotation") == NotImplemented

    def test_cycle_info(self):
        # TODO: throw an error instead
        a = Annotation.from_timeline(42, self.shape, self.shape_timeline)
        assert a.cycle_info() == "Annotation doesn't have a cycle"

    def test_str(self):
        a1 = Annotation(42, self.shape, self.shape_frame_to_label, info="This is info")
        assert str(a1) == 'Annotation type: shape\nThis is info\nTotal frames : 42\n'

    def test_repr(self):
        a1 = Annotation(42, self.shape, self.shape_frame_to_label, info="This is info")
        assert repr(a1) == 'Annotation type: shape\nThis is info\nTotal frames : 42\n'

    def test_from_df(self):
        df = self.shape_cycle.to_df()
        a1 = Annotation.from_df(42, df, is_cycle=True)
        a2 = Annotation.from_cycle(42, self.shape, self.shape_cycle)
        assert a1 == a2

        df = self.shape_timeline.to_df()
        a1 = Annotation.from_df(42, df, is_cycle=False)
        a3 = Annotation.from_timeline(42, self.shape, self.shape_timeline)
        assert a1 != a2
        assert a1 == a3


"""
Tests for the `vodex.core` module.
"""
import json
from pathlib import Path
import pytest
from vodex import *

TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')




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

    def test_get_frame_mapping(self):
        frame_m = FrameManager(self.file_m)
        frame_to_file, frame_in_file = frame_m._get_frame_mapping()

        assert frame_to_file == self.frame_to_file
        assert frame_in_file == self.frame_in_file

    def test_from_dir(self):
        frame_m1 = FrameManager(self.file_m)
        frame_m2 = FrameManager.from_dir(self.data_dir_split)
        assert frame_m1 == frame_m2


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


"""
Tests for the `vodex.сщку` module.
"""
import json
from pathlib import Path
import pytest

from vodex import *

TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')


class TestFileManager:

    data_dir_full = TEST_DATA
    data_dir_split = Path(TEST_DATA, "test_movie")

    full_movie = Path(TEST_DATA, "test_movie.tif")
    file_names = ["mov0.tif", "mov1.tif", "mov2.tif"]
    files = [Path(TEST_DATA, "test_movie", mov) for mov in file_names]

    def test_eq(self):
        file_m1 = FileManager(self.data_dir_split)
        file_m2 = FileManager(self.data_dir_split)
        assert file_m1 == file_m2
        assert file_m2 == file_m1

    def test_find_files(self):
        with pytest.raises(AssertionError):
            FileManager("Duck")
        file_names = FileManager(self.data_dir_split).find_files(".tif")
        assert file_names == self.file_names

    def test_check_files(self):
        file_names = FileManager(self.data_dir_split).check_files(self.file_names)
        assert file_names == self.file_names

    def test_get_frames_per_file(self):
        frames_per_file = FileManager(self.data_dir_split).get_frames_per_file()
        assert frames_per_file == [7, 18, 17]

    def test_state(self):
        pass

    def test_str(self):
        file_m = FileManager(self.data_dir_split)
        print(file_m)

    def test_repr(self):
        pass


class TestTimeLabel:
    shape = Labels("shape", ["c", "s"],
                   state_info={"c": "circle on the screen", "s": "square on the screen"})
    light = Labels("light", ["on", "off"], group_info="Information about the light",
                   state_info={"on": "the intensity of the background is high",
                               "off": "the intensity of the background is low"})
    c_num = Labels("c label", ['c1', 'c2', 'c3'], state_info={'c1': 'written c1', 'c2': 'written c1'})

    def test_eq(self):
        c1 = TimeLabel("c", description="circle on the screen", group="shape")
        c2 = TimeLabel("c", description="circle on the screen", group="shape")
        c3 = TimeLabel("c", description="circle on the screen")
        c4 = TimeLabel("c", group="shape")
        c5 = TimeLabel("c")

        assert c1 == c2
        assert c2 == c1
        # due to no group in c3
        assert c1 != c3
        assert c3 != c1

        assert c1 == c4
        assert c4 == c1
        # due to no group in c5
        assert c1 != c5
        assert c5 != c1
        assert c3 == c5
        assert c5 == c3

        c6 = TimeLabel("c", group="c label")

        assert c1 != c6
        assert c6 != c1
        assert c4 != c6
        assert c6 != c4

        s1 = TimeLabel("s", group="shape")

        assert c1 != s1
        assert s1 != c1
        assert c5 != s1
        assert s1 != c5

        s2 = TimeLabel("s", group="c label")

        assert c1 != s2
        assert s2 != c1
        assert c5 != s2
        assert s2 != c5

    def test_to_dict(self):
        c1 = TimeLabel("c", description="circle on the screen", group="shape")
        c2 = TimeLabel("c", description="circle on the screen")
        c3 = TimeLabel("c", group="shape")

        d1 = {"name": "c", "group": "shape", "description": "circle on the screen"}
        d2 = {"name": "c", "description": "circle on the screen"}
        d3 = {"name": "c", "group": "shape"}

        assert c1.to_dict() == d1
        assert c2.to_dict() == d2
        assert c3.to_dict() == d3

    def test_from_dict(self):
        c1 = TimeLabel("c", description="circle on the screen", group="shape")
        c2 = TimeLabel("c", description="circle on the screen")
        c3 = TimeLabel("c", group="shape")

        d1 = {"name": "c", "group": "shape", "description": "circle on the screen"}
        d2 = {"name": "c", "description": "circle on the screen"}
        d3 = {"name": "c", "group": "shape"}

        assert TimeLabel.from_dict(d1) == c1
        assert TimeLabel.from_dict(d1).to_dict() == d1

        assert TimeLabel.from_dict(d2) == c2
        assert TimeLabel.from_dict(d2).to_dict() == d2

        assert TimeLabel.from_dict(d3) == c3
        assert TimeLabel.from_dict(d3).to_dict() == d3


class TestLabel:

    def test_state(self):
        c = TimeLabel("c", description="circle on the screen", group="shape")
        s = TimeLabel("s", description="square on the screen", group="shape")

        shape = Labels("shape", ["c", "s"],
                       group_info="Information about the shape of a circle/square on the screen",
                       state_info={"c": "circle on the screen", "s": "square on the screen"})
        assert shape.group == "shape"
        assert shape.group_info == "Information about the shape of a circle/square on the screen"
        assert shape.state_names == ["c", "s"]
        assert shape.states == [c, s]
        assert shape.states[0].description == "circle on the screen"
        assert shape.states[1].description == "square on the screen"
        assert shape.c == c
        assert shape.s == s


class TestCycle:
    shape = Labels("shape", ["c", "s"],
                   state_info={"c": "circle on the screen", "s": "square on the screen"})
    per_frame_label_list = [shape.c, shape.c, shape.c, shape.c, shape.c,
                            shape.s, shape.s, shape.s, shape.s, shape.s,
                            shape.s, shape.s, shape.s, shape.s, shape.s,
                            shape.c, shape.c, shape.c, shape.c, shape.c]

    label_per_frame_list = [shape.c, shape.c, shape.c, shape.c, shape.c,
                            shape.s, shape.s, shape.s, shape.s, shape.s,
                            shape.s, shape.s, shape.s, shape.s, shape.s,
                            shape.c, shape.c, shape.c, shape.c, shape.c,  # 20
                            shape.c, shape.c, shape.c, shape.c, shape.c,
                            shape.s, shape.s, shape.s, shape.s, shape.s,
                            shape.s, shape.s, shape.s, shape.s, shape.s,
                            shape.c, shape.c, shape.c, shape.c, shape.c,  # 40
                            shape.c, shape.c]  # 42

    cycle_per_frame_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            2, 2]

    shape_cycle = Cycle([shape.c, shape.s, shape.c], [5, 10, 5])

    def test_eq(self):
        cycle1 = Cycle([self.shape.c, self.shape.s, self.shape.c], [5, 10, 5])
        cycle2 = Cycle([self.shape.c, self.shape.s, self.shape.c], [5, 10, 5])
        cycle3 = Cycle([self.shape.s, self.shape.c, self.shape.s], [5, 10, 5])
        cycle4 = Cycle([self.shape.c, self.shape.s, self.shape.c], [2, 10, 8])

        assert cycle1 == cycle2
        assert cycle2 == cycle1
        assert cycle1 != cycle3
        assert cycle3 != cycle1
        assert cycle1 != cycle4
        assert cycle4 != cycle1

    def test_get_label_per_frame(self):
        per_frame_label_list = self.shape_cycle._get_label_per_frame()
        assert per_frame_label_list == self.per_frame_label_list

    def test_fit_frames(self):
        n_cycles = self.shape_cycle.fit_frames(42)
        assert n_cycles == 3

    def test_fit_labels_to_frames(self):
        label_per_frame_list = self.shape_cycle.fit_labels_to_frames(42)
        assert label_per_frame_list == self.label_per_frame_list

    def test_fit_cycles_to_frames(self):
        cycle_per_frame_list = self.shape_cycle.fit_cycles_to_frames(42)
        assert cycle_per_frame_list == self.cycle_per_frame_list

    def test_to_dict(self):
        d = {'timing': [5, 10, 5], 'label_order': [self.shape.c.to_dict(),
                                                   self.shape.s.to_dict(),
                                                   self.shape.c.to_dict()]}
        assert self.shape_cycle.to_dict() == d
        
    def test_to_json(self):
        j = json.dumps({'timing': [5, 10, 5], 'label_order': [self.shape.c.to_dict(),
                                                              self.shape.s.to_dict(),
                                                              self.shape.c.to_dict()]})
        assert self.shape_cycle.to_json() == j

    def test_from_dict(self):
        d = {'timing': [5, 10, 5], 'label_order': [self.shape.c.to_dict(),
                                                   self.shape.s.to_dict(),
                                                   self.shape.c.to_dict()]}
        assert Cycle.from_dict(d) == self.shape_cycle

    def test_from_json(self):
        j = json.dumps({'timing': [5, 10, 5], 'label_order': [self.shape.c.to_dict(),
                                                              self.shape.s.to_dict(),
                                                              self.shape.c.to_dict()]})
        assert Cycle.from_json(j) == self.shape_cycle


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


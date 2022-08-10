from core import *
# import nose2 as nose
# from nose2.tools import params
import unittest

import tifffile as tif
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from sqlite3 import connect

TEST_DATA = r"D:\Code\repos\vodex\data\test"


def plot_4_frames(f_img, test_function):
    """
    Plots frames. Must be frames number 1,2,41,42
    Used in TestTiffLoader and TestImageLoader
    """
    # checking the exact frames
    plt.figure(figsize=(4, 10), dpi=160)
    plt.imshow(f_img.reshape((800, 200)))
    plt.title(f"{test_function}:\n Must show frames\n1, 2, 41, 42")
    plt.axis('off')
    plt.show()


class TestTiffLoader(unittest.TestCase):
    full_movie = Path(TEST_DATA, "test_movie.tif")
    split_movies = [Path(TEST_DATA, "test_movie", mov) for mov in
                    ["mov0.tif", "mov1.tif", "mov2.tif"]]
    frames_1_2_41_42 = tif.imread(Path(TEST_DATA, "frames_1_2_41_42.tif"))

    # def test_state(self):
    #     loader = TiffLoader(self.full_movie)
    #     self.assertEqual(loader.frame_size, (200, 200))

    def test_eq(self):
        loader1 = TiffLoader(self.full_movie)
        loader2 = TiffLoader(self.full_movie)
        self.assertEqual(loader1, loader2)

    def test_get_frames_in_file(self):
        loader = TiffLoader(self.full_movie)
        n_frames = loader.get_frames_in_file(self.full_movie)
        self.assertEqual(n_frames, 42)

    def test_get_frame_size(self):
        loader = TiffLoader(self.full_movie)
        f_size = loader.get_frame_size(self.full_movie)
        self.assertEqual(f_size, (200, 200))

    def test_get_frame_dtype(self):
        loader = TiffLoader(self.full_movie)
        data_type = loader.get_frame_dtype(self.full_movie)
        self.assertEqual(data_type, np.uint16)

    def test_load_frames_one_file(self):
        loader = TiffLoader(self.full_movie)
        frames = [0, 1, 40, 41]
        print("Must show a progress bar:")
        f_img = loader.load_frames(frames, [self.full_movie] * 4)
        self.assertEqual(f_img.shape, (4, 200, 200))
        print("Must show 'Loading from file' and one file:")
        f_img = loader.load_frames(frames, [self.full_movie] * 4, show_file_names=True)
        self.assertEqual(f_img.shape, (4, 200, 200))
        # checking the exact frames
        # plot_4_frames(f_img, "TiffLoader\ntest_load_frames_one_file")
        self.assertTrue(np.all(np.equal(f_img, self.frames_1_2_41_42)))

    def test_load_frames_many_files(self):
        loader = TiffLoader(self.full_movie)
        frames = [0, 1, 15, 16]
        print("Must show a progress bar:")
        files = [self.split_movies[0], self.split_movies[0],
                 self.split_movies[2], self.split_movies[2]]
        f_img = loader.load_frames(frames, files)
        self.assertEqual(f_img.shape, (4, 200, 200))
        print("Must show 'Loading from file' and two files:")
        f_img = loader.load_frames(frames, files, show_file_names=True)
        self.assertEqual(f_img.shape, (4, 200, 200))
        # checking the exact frames
        # plot_4_frames(f_img, "TiffLoader\ntest_load_frames_many_files")
        self.assertTrue(np.all(np.equal(f_img, self.frames_1_2_41_42)))


class TestImageLoader(unittest.TestCase):
    data_dir_full = TEST_DATA

    full_movie = Path(TEST_DATA, "test_movie.tif")
    split_movies = [Path(TEST_DATA, "test_movie", mov) for mov in
                    ["mov0.tif", "mov1.tif", "mov2.tif"]]

    frames_1_2_41_42 = tif.imread(Path(TEST_DATA, "frames_1_2_41_42.tif"))
    volumes_1_2 = tif.imread(Path(TEST_DATA, "volumes_1_2.tif"))
    half_volumes_1_2 = tif.imread(Path(TEST_DATA, "half_volumes_1_2.tif"))

    def test_eq(self):
        loader1 = ImageLoader(self.full_movie)
        loader2 = ImageLoader(self.full_movie)
        self.assertEqual(loader1, loader2)

    def test_choose_loader(self):
        tif_loader = TiffLoader(self.full_movie)
        loader = ImageLoader(self.full_movie).choose_loader(self.full_movie)
        self.assertEqual(loader, tif_loader)

    def test_get_frames_in_file(self):
        loader = ImageLoader(self.full_movie)
        n_frames = loader.get_frames_in_file(self.full_movie)
        self.assertEqual(n_frames, 42)

    def test_get_frame_size(self):
        loader = ImageLoader(self.full_movie)
        f_size = loader.get_frame_size(self.full_movie)
        self.assertEqual(f_size, (200, 200))

    def test_load_frames_one_file(self):
        loader = ImageLoader(self.full_movie)

        frames = [0, 1, 40, 41]
        files = [self.full_movie] * 4

        print("Must show a progress bar:")
        f_img = loader.load_frames(frames, files)
        self.assertEqual(f_img.shape, (4, 200, 200))

        print("Must show 'Loading from file' and one file:")
        f_img = loader.load_frames(frames, files, show_file_names=True)
        self.assertEqual(f_img.shape, (4, 200, 200))

        # checking the exact frames
        # plot_4_frames(f_img, "ImageLoader\ntest_load_frames_one_file")
        self.assertTrue(np.all(np.equal(f_img, self.frames_1_2_41_42)))

    def test_load_frames_many_files(self):
        loader = ImageLoader(self.full_movie)

        frames = [0, 1, 15, 16]
        files = [self.split_movies[0], self.split_movies[0],
                 self.split_movies[2], self.split_movies[2]]

        print("Must show a progress bar:")
        f_img = loader.load_frames(frames, files)
        self.assertEqual(f_img.shape, (4, 200, 200))

        print("Must show 'Loading from file' and two files:")
        f_img = loader.load_frames(frames, files, show_file_names=True)
        self.assertEqual(f_img.shape, (4, 200, 200))

        # checking the exact frames
        # plot_4_frames(f_img, "ImageLoader\ntest_load_frames_many_files")
        self.assertTrue(np.all(np.equal(f_img, self.frames_1_2_41_42)))

    def test_load_volumes_full(self):
        loader = ImageLoader(self.full_movie)

        frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        volumes = [1] * 10
        volumes.extend([2] * 10)
        files = [self.full_movie] * 20

        v_img = loader.load_volumes(frames, files, volumes)
        self.assertEqual(v_img.shape, (2, 10, 200, 200))
        self.assertTrue(np.all(np.equal(v_img, self.volumes_1_2)))

    def test_load_volumes_half(self):
        loader = ImageLoader(self.full_movie)

        frames = [0, 1, 2, 3, 4,
                  10, 11, 12, 13, 14]
        volumes = [1] * 5
        volumes.extend([2] * 5)
        files = [self.full_movie] * 10

        v_img = loader.load_volumes(frames, files, volumes)
        self.assertEqual(v_img.shape, (2, 5, 200, 200))
        self.assertTrue(np.all(np.equal(v_img, self.half_volumes_1_2)))

        # now let's make sure it breaks when we ask for different number of slices per volume
        volumes = [1] * 6
        volumes.extend([2] * 4)
        files = [self.full_movie] * 10
        with self.assertRaises(AssertionError):
            loader.load_volumes(frames, files, volumes)


class TestFileManager(unittest.TestCase):
    data_dir_full = TEST_DATA
    data_dir_split = Path(TEST_DATA, "test_movie")

    full_movie = Path(TEST_DATA, "test_movie.tif")
    file_names = ["mov0.tif", "mov1.tif", "mov2.tif"]
    files = [Path(TEST_DATA, "test_movie", mov) for mov in file_names]

    def test_eq(self):
        file_m1 = FileManager(self.data_dir_split)
        file_m2 = FileManager(self.data_dir_split)
        self.assertEqual(file_m1, file_m2)

    def test_find_files(self):
        with self.assertRaises(AssertionError):
            FileManager("Duck")
        file_names = FileManager(self.data_dir_split).find_files(".tif")
        self.assertEqual(file_names, self.file_names)

    def test_check_files(self):
        files, file_names = FileManager(self.data_dir_split).check_files(self.file_names)
        self.assertEqual(files, self.files)
        self.assertEqual(file_names, self.file_names)

    def test_get_frames_per_file(self):
        frames_per_file = FileManager(self.data_dir_split).get_frames_per_file()
        self.assertEqual(frames_per_file, [7, 18, 17])

    def test_state(self):
        pass

    def test_str(self):
        pass

    def test_repr(self):
        pass


class TestTimeLabel(unittest.TestCase):
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

        self.assertEqual(c1, c2)
        # due to no group in c3
        self.assertNotEqual(c1, c3)
        self.assertEqual(c1, c4)
        # due to no group in c5
        self.assertNotEqual(c1, c5)
        self.assertEqual(c3, c5)

        c6 = TimeLabel("c", group="c label")

        self.assertNotEqual(c1, c6)
        self.assertNotEqual(c4, c6)

        s1 = TimeLabel("s", group="shape")

        self.assertNotEqual(c1, s1)
        self.assertNotEqual(c5, s1)

        s2 = TimeLabel("s", group="c label")

        self.assertNotEqual(c1, s2)
        self.assertNotEqual(c5, s2)

    def test_to_dict(self):
        c1 = TimeLabel("c", description="circle on the screen", group="shape")
        c2 = TimeLabel("c", description="circle on the screen")
        c3 = TimeLabel("c", group="shape")

        d1 = {"name": "c", "group": "shape", "description": "circle on the screen"}
        d2 = {"name": "c", "description": "circle on the screen"}
        d3 = {"name": "c", "group": "shape"}

        self.assertEqual(c1.to_dict(), d1)
        self.assertEqual(c2.to_dict(), d2)
        self.assertEqual(c3.to_dict(), d3)

    def test_from_dict(self):
        c1 = TimeLabel("c", description="circle on the screen", group="shape")
        c2 = TimeLabel("c", description="circle on the screen")
        c3 = TimeLabel("c", group="shape")

        d1 = {"name": "c", "group": "shape", "description": "circle on the screen"}
        d2 = {"name": "c", "description": "circle on the screen"}
        d3 = {"name": "c", "group": "shape"}

        self.assertEqual(TimeLabel.from_dict(d1), c1)
        self.assertEqual(TimeLabel.from_dict(d1).to_dict(), d1)

        self.assertEqual(TimeLabel.from_dict(d2), c2)
        self.assertEqual(TimeLabel.from_dict(d2).to_dict(), d2)

        self.assertEqual(TimeLabel.from_dict(d3), c3)
        self.assertEqual(TimeLabel.from_dict(d3).to_dict(), d3)

    # def test_state(self):
    #     pass
    #
    # def test_str(self):
    #     pass
    #
    # def test_repr(self):
    #     pass


class TestLabel(unittest.TestCase):

    def test_state(self):
        c = TimeLabel("c", description="circle on the screen", group="shape")
        s = TimeLabel("s", description="square on the screen", group="shape")

        shape = Labels("shape", ["c", "s"],
                       group_info="Information about the shape of a circle/square on the screen",
                       state_info={"c": "circle on the screen", "s": "square on the screen"})
        self.assertEqual(shape.group, "shape")
        self.assertEqual(shape.group_info, "Information about the shape of a circle/square on the screen")
        self.assertEqual(shape.state_names, ["c", "s"])
        self.assertEqual(shape.states, [c, s])
        self.assertEqual(shape.states[0].description, "circle on the screen")
        self.assertEqual(shape.states[1].description, "square on the screen")
        self.assertEqual(shape.c, c)
        self.assertEqual(shape.s, s)

    # def test_str(self):
    #     pass
    #
    # def test_repr(self):
    #     pass


class TestCycle(unittest.TestCase):
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

        self.assertEqual(cycle1, cycle2)
        self.assertNotEqual(cycle1, cycle3)
        self.assertNotEqual(cycle1, cycle4)

    def test_get_label_per_frame(self):
        per_frame_label_list = self.shape_cycle.get_label_per_frame()
        self.assertEqual(per_frame_label_list, self.per_frame_label_list)

    def test_fit_frames(self):
        n_cycles = self.shape_cycle.fit_frames(42)
        self.assertEqual(n_cycles, 3)

    def test_fit_labels_to_frames(self):
        label_per_frame_list = self.shape_cycle.fit_labels_to_frames(42)
        self.assertEqual(label_per_frame_list, self.label_per_frame_list)

    def test_fit_cycles_to_frames(self):
        cycle_per_frame_list = self.shape_cycle.fit_cycles_to_frames(42)
        self.assertEqual(cycle_per_frame_list, self.cycle_per_frame_list)

    def test_to_dict(self):
        d = {'timing': [5, 10, 5], 'label_order': [self.shape.c.to_dict(),
                                                   self.shape.s.to_dict(),
                                                   self.shape.c.to_dict()]}
        self.assertEqual(self.shape_cycle.to_dict(), d)

    def test_to_json(self):
        j = json.dumps({'timing': [5, 10, 5], 'label_order': [self.shape.c.to_dict(),
                                                              self.shape.s.to_dict(),
                                                              self.shape.c.to_dict()]})
        self.assertEqual(self.shape_cycle.to_json(), j)

    def test_from_dict(self):
        d = {'timing': [5, 10, 5], 'label_order': [self.shape.c.to_dict(),
                                                   self.shape.s.to_dict(),
                                                   self.shape.c.to_dict()]}
        self.assertEqual(Cycle.from_dict(d), self.shape_cycle)

    def test_from_json(self):
        j = json.dumps({'timing': [5, 10, 5], 'label_order': [self.shape.c.to_dict(),
                                                              self.shape.s.to_dict(),
                                                              self.shape.c.to_dict()]})
        self.assertEqual(Cycle.from_json(j), self.shape_cycle)

    # def test_state(self):
    #     pass
    #
    # def test_str(self):
    #     pass
    #
    # def test_repr(self):
    #     pass


class TestFrameManager(unittest.TestCase):
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
        self.assertEqual(frame_m1, frame_m2)

    def test_get_frame_mapping(self):
        frame_m = FrameManager(self.file_m)
        frame_to_file, frame_in_file = frame_m.get_frame_mapping()

        self.assertEqual(frame_to_file, self.frame_to_file)
        self.assertEqual(frame_in_file, self.frame_in_file)

    def test_from_dir(self):
        frame_m1 = FrameManager(self.file_m)
        frame_m2 = FrameManager.from_dir(self.data_dir_split)
        self.assertEqual(frame_m1, frame_m2)

    # def test_state(self):
    #     pass
    #
    # def test_str(self):
    #     pass
    #
    # def test_repr(self):
    #     pass


class TestVolumeManager(unittest.TestCase):
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
        frame_to_z = self.volume_m.get_frames_to_z_mapping()
        self.assertEqual(frame_to_z, self.frame_to_z)

    def test_get_frames_to_volumes_mapping(self):
        frame_to_vol = self.volume_m.get_frames_to_volumes_mapping()
        self.assertEqual(frame_to_vol, self.frame_to_vol)

    # def test_from_dir(self):
    #     pass
    #
    # def test_state(self):
    #     pass
    #
    # def test_str(self):
    #     pass
    #
    # def test_repr(self):
    #     pass


class TestAnnotation(unittest.TestCase):
    shape = Labels("shape", ["c", "s"],
                   state_info={"c": "circle on the screen", "s": "square on the screen"})
    shape_cycle = Cycle([shape.c, shape.s, shape.c], [5, 10, 5])

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

    def test_from_cycle(self):
        a1 = Annotation(42, self.shape, self.shape_frame_to_label)
        a2 = Annotation.from_cycle(42, self.shape, self.shape_cycle)

        self.assertEqual(a1.frame_to_label, a2.frame_to_label)
        self.assertEqual(a1.n_frames, a2.n_frames)
        self.assertEqual(a1.labels, a2.labels)
        self.assertEqual(a1.name, a2.name)

        self.assertTrue(a1.cycle is None)
        self.assertEqual(a2.cycle, self.shape_cycle)
        self.assertEqual(a2.frame_to_cycle, self.frame_to_cycle)

    # def test_state(self):
    #     pass

    # def test_str(self):
    #     pass
    #
    # def test_repr(self):
    #     pass


class TestExperiment(unittest.TestCase):

    def test_state(self):
        pass

    def test_create(self):
        pass

    def test_choose_frames(self):
        pass

    def test_choose_volumes(self):
        pass
#

#
# class TestDbManager(unittest.TestCase):
#
#     def test_state(self):
#         pass
#
#     def list_tables(self):
#         pass
#
#     def table_as_df(self):
#         pass
#
#     def get_n_frames(self):
#         pass
#
#     def save(self):
#         pass
#
#     def empty(self):
#         pass
#
#     def load(self):
#         pass
#
#     def populate(self):
#         pass
#
#     def _create_tables(self):
#         pass
#
#     def _populate_Options(self):
#         pass
#
#     def _populate_Files(self):
#         pass
#
#     def _populate_Frames(self):
#         pass
#
#     def _populate_Volumes(self):
#         pass
#
#     def _populate_AnnotationTypes(self):
#         pass
#
#     def _populate_AnnotationTypeLabels(self):
#         pass
#
#     def _populate_Annotations(self):
#         """
#         select * from tableA
#         minus
#         select * from tableB
#         """
#
#     def _populate_Cycles(self):
#         pass
#
#     def _populate_CycleIterations(self):
#         pass

if __name__ == "__main__":
    unittest.main()

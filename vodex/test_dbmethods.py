from core import *
from dbmethods import *

import unittest
from pathlib import Path

TEST_DATA = r"D:\Code\repos\vodex\data\test"


class TestDbWriter(unittest.TestCase):
    data_dir_split = Path(TEST_DATA, "test_movie")

    shape = Labels("shape", ["c", "s"],
                   state_info={"c": "circle on the screen", "s": "square on the screen"})
    light = Labels("light", ["on", "off"], group_info="Information about the light",
                   state_info={"on": "the intensity of the background is high",
                               "off": "the intensity of the background is low"})
    cnum = Labels("c label", ['c1', 'c2', 'c3'], state_info={'c1': 'written c1', 'c2': 'written c1'})

    shape_cycle = Cycle([shape.c, shape.s, shape.c], [5, 10, 5])
    cnum_cycle = Cycle([cnum.c1, cnum.c2, cnum.c3], [10, 10, 10])
    light_order = [light.off, light.on, light.off]
    light_timing = [10, 20, 12]

    shape_an = Annotation.from_cycle(42, shape, shape_cycle)
    cnum_an = Annotation.from_cycle(42, cnum, cnum_cycle)
    light_an = Annotation.from_timing(42, light, light_order, light_timing)

    def test_save(self):
        # not sure how to test this maybe just look at the saved database for now? >_<
        pass

    def test_create(self):
        # not sure how to test this
        db = DbWriter.create()
        db.connection.close()

    def test_load(self):
        # not sure how to test this
        db = DbWriter.load(Path(TEST_DATA, "test.db"))
        db.connection.close()

    def test_populate(self):
        # TODO: test that lists for the db are true int all the time !!!!
        # TODO: test what happens when volume manager and annotation have different number of frames
        volume_m = VolumeManager.from_dir(self.data_dir_split, 10, fgf=0)
        db = DbWriter.create()
        db.populate(volumes=volume_m, annotations=[self.shape_an, self.cnum_an, self.light_an])
        db.save(Path(TEST_DATA, "test.db"))
        db.connection.close()


class TestDbReader(unittest.TestCase):

    def test_load(self):
        # not sure how to test this
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        db.connection.close()

    def test_get_frames_per_volumes(self):
        frames_vol01 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        frames_vol0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        db = DbReader.load(Path(TEST_DATA, "test.db"))

        frame_ids = db.get_frames_per_volumes([0])
        self.assertEqual(frame_ids, frames_vol0)

        frame_ids = db.get_frames_per_volumes([0, 1])
        self.assertEqual(frame_ids, frames_vol01)

        # duplicating doesn't change the frames
        frame_ids = db.get_frames_per_volumes([0, 1, 1])
        self.assertEqual(frame_ids, frames_vol01)

        db.connection.close()

    def test_get_and_frames_per_annotations(self):
        frames_cond1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        frames_cond2 = [6, 7, 8, 9, 10, 31, 32, 33, 34, 35]

        cond1 = [("c label", "c1")]
        cond2 = [("c label", "c1"), ("shape", "s")]

        db = DbReader.load(Path(TEST_DATA, "test.db"))

        frame_ids = db.get_and_frames_per_annotations(cond1)
        self.assertEqual(frame_ids, frames_cond1)

        frame_ids = db.get_and_frames_per_annotations(cond2)
        self.assertEqual(frame_ids, frames_cond2)

        db.connection.close()

    def test_get_or_frames_per_annotations(self):
        frames_cond1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        frames_cond2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

        cond1 = [("c label", "c1")]
        cond2 = [("c label", "c1"), ("shape", "s")]

        db = DbReader.load(Path(TEST_DATA, "test.db"))

        frame_ids = db.get_or_frames_per_annotations(cond1)
        self.assertEqual(frame_ids, frames_cond1)

        frame_ids = db.get_or_frames_per_annotations(cond2)
        self.assertEqual(frame_ids, frames_cond2)

        db.connection.close()

    def test_prepare_frames_for_loading(self):
        frames1 = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 41, 42]
        data_dir1 = "D:/Code/repos/vodex/data/test/test_movie"
        file_names1 = ["mov0.tif", "mov1.tif", "mov2.tif"]
        files1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3]
        frame_in_file1 = [0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 13, 14, 15, 16]
        volumes1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, -2, -2]

        db = DbReader.load(Path(TEST_DATA, "test.db"))
        data_dir, file_names, files, frame_in_file, volumes = db.prepare_frames_for_loading(frames1)
        db.connection.close()

        self.assertEqual(data_dir1, data_dir)
        self.assertEqual(file_names1, file_names)
        self.assertEqual(files1, files)
        self.assertEqual(frame_in_file1, frame_in_file)
        self.assertEqual(volumes1, volumes)

    def test__get_AnnotationLabelId(self):
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        label_id = db._get_AnnotationLabelId(("shape", "c"))
        db.connection.close()

        self.assertEqual(label_id, 1)

    def test__get_VolumeId_from_Volumes(self):
        frames1 = [1, 2, 3, 11, 12, 13]
        frames2 = [11, 12, 13, 1, 2, 3]

        volumes1 = [0, 0, 0, 1, 1, 1]
        volumes2 = [1, 1, 1, 0, 0, 0]

        db = DbReader.load(Path(TEST_DATA, "test.db"))

        volume_ids = db._get_VolumeId_from_Volumes(frames1)
        self.assertEqual(volume_ids, volumes1)

        # does not preserve order
        volume_ids = db._get_VolumeId_from_Volumes(frames2)
        self.assertNotEqual(volume_ids, volumes2)

        db.connection.close()

    def test_choose_frames_per_slices(self):
        frames1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12,
                   21, 22, 23]

        frames2 = [2, 6, 12, 16, 17, 22, 26, 27]

        db = DbReader.load(Path(TEST_DATA, "test.db"))

        chosen_frames = db.choose_frames_per_slices(frames1, [0, 1, 2])
        self.assertEqual(chosen_frames, [1, 2, 3, 21, 22, 23])

        chosen_frames = db.choose_frames_per_slices(frames2, [1, 5, 6])
        self.assertEqual(chosen_frames, [12, 16, 17, 22, 26, 27])

        chosen_frames = db.choose_frames_per_slices(frames2, [5, 1, 6])
        self.assertEqual(chosen_frames, [12, 16, 17, 22, 26, 27])

        db.connection.close()

    def test_choose_full_volumes(self):
        frames1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23]

        chosen_frames1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        chosen_volumes1 = [0,1]

        frames2 = [2, 6, 12, 16, 17, 22, 26, 27]
        chosen_frames2 = []
        chosen_volumes2 = []

        db = DbReader.load(Path(TEST_DATA, "test.db"))

        chosen_volumes, chosen_frames = db.choose_full_volumes(frames1)
        self.assertEqual(chosen_frames1, chosen_frames)
        self.assertEqual(chosen_volumes1, chosen_volumes)

        chosen_volumes, chosen_frames = db.choose_full_volumes(frames2)
        self.assertEqual(chosen_frames2, chosen_frames)
        self.assertEqual(chosen_volumes2, chosen_volumes)

        db.connection.close()


# class TestDbExporter(unittest.TestCase):
#
#     def test_table_as_df(self):
#         # not sure how to test this
#         db = DbExporter.load(Path(TEST_DATA, "test.db"))
#         df = db.table_as_df("AnnotationTypeLabels")
#         db.connection.close()


if __name__ == "__main__":
    # TODO: test that lists for the db are true int all the time !!!!
    unittest.main()

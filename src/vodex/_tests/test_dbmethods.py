"""
Tests for the `vodex.dbmethods` module.
"""
import pytest
from vodex import *
from pathlib import Path

TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')

class TestDbReader:

    def test_get_n_frames(self):
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        n_frames = db.get_n_frames()
        assert 42 == n_frames

    def test_get_fpv(self):
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        fpv = db.get_fpv()
        assert 10 == fpv

    def test_get_fgf(self):
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        fpv = db.get_fgf()
        assert 0 == fpv

    def test_get_file_names(self):
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        file_names = db.get_file_names()
        assert ['mov0.tif', 'mov1.tif', 'mov2.tif'] == file_names

    def test_get_frames_per_file(self):
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        frames_per_file = db.get_frames_per_file()
        assert [7, 18, 17] == frames_per_file

    def test_get_data_dir(self):
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        data_dir = db.get_data_dir()
        assert Path(TEST_DATA, 'test_movie').samefile(data_dir)

    def test_get_options(self):
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        options = db.get_options()

        assert options['frames_per_volume'] == '10'
        assert options['num_head_frames'] == '0'
        assert options['num_tail_frames'] == '2'
        assert options['num_full_volumes'] == '4'
        assert Path(TEST_DATA, 'test_movie').samefile(options['data_dir'])

    def test_load(self):
        # not sure how to test this
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        db.connection.close()

    def test_get_volume_list(self):
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        volume_ids = db.get_volume_list()
        assert [-2, 0, 1, 2, 3] == volume_ids

    def test_get_frames_per_volumes(self):
        frames_vol01 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        frames_vol0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        db = DbReader.load(Path(TEST_DATA, "test.db"))

        frame_ids = db.get_frames_per_volumes([0])
        assert frame_ids == frames_vol0

        frame_ids = db.get_frames_per_volumes([0, 1])
        assert frame_ids == frames_vol01

        # duplicating doesn't change the frames
        frame_ids = db.get_frames_per_volumes([0, 1, 1])
        assert frame_ids == frames_vol01

        db.connection.close()

    def test_get_and_frames_per_annotations(self):
        frames_cond1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        frames_cond2 = [6, 7, 8, 9, 10, 31, 32, 33, 34, 35]

        cond1 = [("c label", "c1")]
        cond2 = [("c label", "c1"), ("shape", "s")]

        db = DbReader.load(Path(TEST_DATA, "test.db"))

        frame_ids = db.get_and_frames_per_annotations(cond1)
        assert frame_ids == frames_cond1

        frame_ids = db.get_and_frames_per_annotations(cond2)
        assert frame_ids == frames_cond2

        db.connection.close()

    def test_get_or_frames_per_annotations(self):
        frames_cond1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        frames_cond2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

        cond1 = [("c label", "c1")]
        cond2 = [("c label", "c1"), ("shape", "s")]

        db = DbReader.load(Path(TEST_DATA, "test.db"))

        frame_ids = db.get_or_frames_per_annotations(cond1)
        assert frame_ids == frames_cond1

        frame_ids = db.get_or_frames_per_annotations(cond2)
        assert frame_ids == frames_cond2

        db.connection.close()

    def test_prepare_frames_for_loading(self):
        frames1 = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 41, 42]
        data_dir1 = Path(TEST_DATA, 'test_movie')
        file_names1 = ["mov0.tif", "mov1.tif", "mov2.tif"]
        files1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3]
        frame_in_file1 = [0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 13, 14, 15, 16]
        volumes1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, -2, -2]

        db = DbReader.load(Path(TEST_DATA, "test.db"))
        data_dir, file_names, files, frame_in_file, volumes = db.prepare_frames_for_loading(frames1)
        db.connection.close()

        assert data_dir1.samefile(data_dir)
        assert file_names1 == file_names
        assert files1 == files
        assert frame_in_file1 == frame_in_file
        assert volumes1 == volumes

    def test__Id_from_AnnotationTypeLabels(self):
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        label_id = db._get_Id_from_AnnotationTypeLabels(("shape", "c"))
        db.connection.close()

        assert label_id == 1

    def test__Ids_from_AnnotationTypeLabels(self):
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        label_ids = db._get_Ids_from_AnnotationTypeLabels("light")
        db.connection.close()
        assert label_ids == [(7,), (6,)]

    def test__get_AnnotationTypeLabelId_from_Annotations(self):
        db = DbReader.load(Path(TEST_DATA, "test.db"))
        label_ids = db.get_AnnotationTypeLabelId_from_Annotations("shape")
        db.connection.close()
        shape_ids = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
        assert label_ids == shape_ids

    def test__get_VolumeId_from_Volumes(self):
        frames1 = [1, 2, 3, 11, 12, 13]
        frames2 = [11, 12, 13, 1, 2, 3]

        volumes1 = [0, 0, 0, 1, 1, 1]
        volumes2 = [1, 1, 1, 0, 0, 0]

        db = DbReader.load(Path(TEST_DATA, "test.db"))

        volume_ids = db._get_VolumeId_from_Volumes(frames1)
        assert volume_ids == volumes1

        # does not preserve order
        volume_ids = db._get_VolumeId_from_Volumes(frames2)
        assert volume_ids != volumes2

        db.connection.close()

    def test_choose_frames_per_slices(self):
        frames1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12,
                   21, 22, 23]

        frames2 = [2, 6, 12, 16, 17, 22, 26, 27]

        db = DbReader.load(Path(TEST_DATA, "test.db"))

        chosen_frames = db.choose_frames_per_slices(frames1, [0, 1, 2])
        assert chosen_frames == [1, 2, 3, 21, 22, 23]

        chosen_frames = db.choose_frames_per_slices(frames2, [1, 5, 6])
        assert chosen_frames == [12, 16, 17, 22, 26, 27]

        chosen_frames = db.choose_frames_per_slices(frames2, [5, 1, 6])
        assert chosen_frames == [12, 16, 17, 22, 26, 27]

        db.connection.close()




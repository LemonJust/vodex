import pytest
from pathlib import Path
from sqlite3 import connect, Connection
from vodex import DbWriter, DbReader, Annotation, Labels, Cycle, Timeline

TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')
# test data movie, where all the data is split into 3 files
SPLIT_MOVIE_DIR = Path(TEST_DATA, "test_movie")
SPLIT_MOVIE_NAMES = ["mov0.tif", "mov1.tif", "mov2.tif"]
SPLIT_MOVIE_FRAMES = [7, 18, 17]
# TODO: test on a full movie too ?

TEST_DB = Path(TEST_DATA, "test.db")


# annotations to create an experiment
def prepare_annotations():
    shape = Labels("shape", ["c", "s"],
                   state_info={"c": "circle on the screen", "s": "square on the screen"})
    light = Labels("light", ["on", "off"], group_info="Information about the light",
                   state_info={"on": "the intensity of the background is high",
                               "off": "the intensity of the background is low"})
    cnum = Labels("c label", ['c1', 'c2', 'c3'], state_info={'c1': 'written c1', 'c2': 'written c2'})

    shape_cycle = Cycle([shape.c, shape.s, shape.c], [5, 10, 5])
    cnum_cycle = Cycle([cnum.c1, cnum.c2, cnum.c3], [10, 10, 10])
    light_tml = Timeline([light.off, light.on, light.off], [10, 20, 12])

    shape_an = Annotation.from_cycle(42, shape, shape_cycle)
    cnum_an = Annotation.from_cycle(42, cnum, cnum_cycle)
    light_an = Annotation.from_timeline(42, light, light_tml)

    return shape_an, cnum_an, light_an


SHAPE_AN, CNUM_AN, LIGHT_AN = prepare_annotations()


@pytest.fixture
def db_reader_empty() -> DbReader:
    dbwriter = DbWriter.create()
    return DbReader(dbwriter.connection)


@pytest.fixture
def db_reader() -> DbReader:
    return DbReader.load(TEST_DB)


def test_get_n_frames(db_reader, db_reader_empty):
    # Get n frames from the database
    assert db_reader.get_n_frames() == 42
    with pytest.raises(Exception) as e:
        db_reader_empty.get_n_frames()
    assert str(e.value) == "no such table: Frames"


def test_get_data_dir(db_reader, db_reader_empty):
    # Get n frames from the database
    assert db_reader.get_data_dir() == SPLIT_MOVIE_DIR.as_posix()
    with pytest.raises(Exception) as e:
        db_reader_empty.get_data_dir()
    assert str(e.value) == "no such table: Options"


def test_get_fpv(db_reader, db_reader_empty):
    assert db_reader.get_fpv() == 10
    with pytest.raises(Exception) as e:
        db_reader_empty.get_fpv()
    assert str(e.value) == "no such table: Options"


def test_get_fgf(db_reader, db_reader_empty):
    assert db_reader.get_fgf() == 0
    with pytest.raises(Exception) as e:
        db_reader_empty.get_fgf()
    assert str(e.value) == "no such table: Options"


def test_get_options(db_reader, db_reader_empty):
    options = db_reader.get_options()
    assert options['frames_per_volume'] == '10'
    assert options['num_head_frames'] == '0'
    assert options['num_tail_frames'] == '2'
    assert options['num_full_volumes'] == '4'
    assert Path(TEST_DATA, 'test_movie').samefile(options['data_dir'])
    with pytest.raises(Exception) as e:
        db_reader_empty.get_options()
    assert str(e.value) == "no such table: Options"


def test_get_file_names(db_reader, db_reader_empty):
    assert db_reader.get_file_names() == ['mov0.tif', 'mov1.tif', 'mov2.tif']
    with pytest.raises(Exception) as e:
        db_reader_empty.get_file_names()
    assert str(e.value) == "no such table: Files"


def test_get_frames_per_file(db_reader, db_reader_empty):
    assert db_reader.get_frames_per_file() == SPLIT_MOVIE_FRAMES
    with pytest.raises(Exception) as e:
        db_reader_empty.get_frames_per_file()
    assert str(e.value) == "no such table: Files"


def test_get_volume_list(db_reader, db_reader_empty):
    assert db_reader.get_volume_list() == [-2, 0, 1, 2, 3]
    with pytest.raises(Exception) as e:
        db_reader_empty.get_volume_list()
    assert str(e.value) == "no such table: Volumes"


def test_load(db_reader, db_reader_empty):
    # Assert that the loaded db_reader object is created and its connection is not None
    assert db_reader is not None
    assert db_reader.connection is not None
    assert db_reader_empty is not None
    assert db_reader_empty.connection is not None
    # TODO: test that it's in memory


def test_choose_full_volumes(db_reader, db_reader_empty):
    # can get full volumes
    frames1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               21, 22, 23]
    chosen_frames1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    chosen_volumes1 = [0, 1]
    # can't get full volumes
    frames2 = [2, 6, 12, 16, 17, 22, 26, 27]
    chosen_frames2 = []
    chosen_volumes2 = []

    chosen_volumes, chosen_frames = db_reader.choose_full_volumes(frames1)
    assert chosen_frames1 == chosen_frames
    assert chosen_volumes1 == chosen_volumes

    chosen_volumes, chosen_frames = db_reader.choose_full_volumes(frames2)
    assert chosen_frames == chosen_frames2
    assert chosen_volumes == chosen_volumes2

    with pytest.raises(Exception) as e:
        db_reader_empty.choose_full_volumes(frames1)
    assert str(e.value) == "no such table: Options"


def test_choose_frames_per_slices(db_reader, db_reader_empty):
    frames1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12,
               21, 22, 23]
    frames2 = [2, 6, 12, 16, 17, 22, 26, 27]

    chosen_frames = db_reader.choose_frames_per_slices(frames1, [0, 1, 2])
    assert chosen_frames == [1, 2, 3, 21, 22, 23]

    chosen_frames = db_reader.choose_frames_per_slices(frames2, [1, 5, 6])
    assert chosen_frames == [12, 16, 17, 22, 26, 27]

    chosen_frames = db_reader.choose_frames_per_slices(frames2, [5, 1, 6])
    assert chosen_frames == [12, 16, 17, 22, 26, 27]

    with pytest.raises(Exception) as e:
        db_reader_empty.choose_frames_per_slices(frames1, [0, 1, 2])
    assert str(e.value) == "no such table: Volumes"


def test_prepare_frames_for_loading(db_reader, db_reader_empty):
    frames = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 41, 42]
    data_dir, file_names, files, frame_in_file, volumes = db_reader.prepare_frames_for_loading(frames)

    assert SPLIT_MOVIE_DIR.samefile(data_dir)
    assert file_names == SPLIT_MOVIE_NAMES
    assert files == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3]
    assert frame_in_file == [0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 13, 14, 15, 16]
    assert volumes == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, -2, -2]

    with pytest.raises(Exception) as e:
        db_reader_empty.prepare_frames_for_loading(frames)
    assert str(e.value) == "no such table: Options"


def test_get_frames_per_volumes(db_reader, db_reader_empty):
    assert db_reader.get_frames_per_volumes([0]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert db_reader.get_frames_per_volumes([0, 1]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # duplicating doesn't change the frames
    assert db_reader.get_frames_per_volumes([0, 1, 1]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                           11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    with pytest.raises(Exception) as e:
        db_reader_empty.get_frames_per_volumes([0])
    assert str(e.value) == "no such table: Volumes"


def test_get_and_frames_per_annotations(db_reader, db_reader_empty):
    assert db_reader.get_and_frames_per_annotations([("c label", "c1")]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                                             31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    assert db_reader.get_and_frames_per_annotations([("c label", "c1"),
                                                     ("shape", "s")]) == [6, 7, 8, 9, 10, 31, 32, 33, 34, 35]
    with pytest.raises(Exception) as e:
        db_reader_empty.get_and_frames_per_annotations([("c label", "c1")])
    assert str(e.value) == "no such table: AnnotationTypeLabels"


def test_get_or_frames_per_annotations(db_reader, db_reader_empty):
    assert db_reader.get_or_frames_per_annotations([("c label", "c1")]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                                            31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    assert db_reader.get_or_frames_per_annotations([("c label", "c1"),
                                                    ("shape", "s")]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                                         11, 12, 13, 14, 15,
                                                                         26, 27, 28, 29, 30,
                                                                         31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    with pytest.raises(Exception) as e:
        db_reader_empty.get_or_frames_per_annotations([("c label", "c1")])
    assert str(e.value) == "no such table: AnnotationTypeLabels"


def test_get_conditionIds_per_cycle_per_volumes(db_reader, db_reader_empty):
    volume_ids, condition_ids, count = db_reader.get_conditionIds_per_cycle_per_volumes("shape")
    assert volume_ids == [0, 0, 1, 1]
    assert condition_ids == [1, 2, 1, 2]
    assert count == [5, 5, 5, 5]

    volume_ids, condition_ids, count = db_reader.get_conditionIds_per_cycle_per_volumes("c label")
    assert volume_ids == [0, 1, 2]
    assert condition_ids == [3, 4, 5]
    assert count == [10, 10, 10]

    with pytest.raises(Exception) as e:
        db_reader.get_conditionIds_per_cycle_per_volumes("light")
    assert str(e.value) == "No Cycle for light"

    with pytest.raises(Exception) as e:
        db_reader_empty.get_conditionIds_per_cycle_per_volumes("shape")
    assert str(e.value) == "no such table: Cycles"


def test_get_conditionIds_per_cycle_per_frame(db_reader, db_reader_empty):
    frame_ids, condition_ids = db_reader.get_conditionIds_per_cycle_per_frame("shape")
    assert frame_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    assert condition_ids == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]

    frame_ids, condition_ids = db_reader.get_conditionIds_per_cycle_per_frame("c label")
    assert frame_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    assert condition_ids == [3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                             4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    with pytest.raises(Exception) as e:
        db_reader.get_conditionIds_per_cycle_per_frame("light")
    assert str(e.value) == "No Cycle for light"

    with pytest.raises(Exception) as e:
        db_reader_empty.get_conditionIds_per_cycle_per_frame("shape")
    assert str(e.value) == "no such table: Cycles"


def test_get_cycleIterations_per_volumes(db_reader, db_reader_empty):
    volume_ids, cycle_its, count = db_reader.get_cycleIterations_per_volumes("shape")
    assert volume_ids == [-2, 0, 1, 2, 3]
    assert cycle_its == [2, 0, 0, 1, 1]
    assert count == [2, 10, 10, 10, 10]

    volume_ids, cycle_its, count = db_reader.get_cycleIterations_per_volumes("c label")
    assert volume_ids == [-2, 0, 1, 2, 3]
    assert cycle_its == [1, 0, 0, 0, 1]
    assert count == [2, 10, 10, 10, 10]

    with pytest.raises(Exception) as e:
        db_reader.get_cycleIterations_per_volumes("light")
    assert str(e.value) == "No Cycle for light"

    with pytest.raises(Exception) as e:
        db_reader_empty.get_cycleIterations_per_volumes("shape")
    assert str(e.value) == "no such table: Cycles"


def test_get_cycleIterations_per_frames(db_reader, db_reader_empty):
    frame_ids, cycle_its = db_reader.get_cycleIterations_per_frame("shape")
    assert frame_ids == list(range(1, 43))
    # 20 times cycle 0, 20 times cycle 1 and 2 times cycle 2:
    assert cycle_its == [ic for ic, k in zip([0, 1, 2], [20, 20, 2]) for i in range(k)]

    frame_ids, cycle_its = db_reader.get_cycleIterations_per_frame("c label")
    assert frame_ids == list(range(1, 43))
    # 30 times cycle 0, 12 times cycle 1
    assert cycle_its == [ic for ic, k in zip([0, 1], [30, 12]) for i in range(k)]

    with pytest.raises(Exception) as e:
        db_reader.get_cycleIterations_per_frame("light")
    assert str(e.value) == "No Cycle for light"

    with pytest.raises(Exception) as e:
        db_reader_empty.get_cycleIterations_per_frame("shape")
    assert str(e.value) == "no such table: Cycles"


def test_get_Names_from_AnnotationTypes(db_reader, db_reader_empty):
    assert db_reader.get_Names_from_AnnotationTypes() == ["shape", "c label", "light"]
    with pytest.raises(Exception) as e:
        db_reader_empty.get_Names_from_AnnotationTypes()
    assert str(e.value) == "no such table: AnnotationTypes"


def test_get_Structure_from_Cycle(db_reader, db_reader_empty):
    assert db_reader.get_Structure_from_Cycle(
        "shape") == '{"timing": [5, 10, 5], "label_order": [{"name": "c", "group": "shape", "description": "circle on the screen"}, {"name": "s", "group": "shape", "description": "square on the screen"}, {"name": "c", "group": "shape", "description": "circle on the screen"}]}'
    assert db_reader.get_Structure_from_Cycle("light") is None
    with pytest.raises(Exception) as e:
        db_reader_empty.get_Structure_from_Cycle("shape")
    assert str(e.value) == "no such table: Cycles"


def test_get_Name_and_Description_from_AnnotationTypeLabels(db_reader, db_reader_empty):
    names, descriptions = db_reader.get_Name_and_Description_from_AnnotationTypeLabels("shape")
    assert names == ['c', 's']
    assert descriptions == {'c': 'circle on the screen', 's': 'square on the screen'}

    names, descriptions = db_reader.get_Name_and_Description_from_AnnotationTypeLabels("light")
    assert names == ['off', 'on']
    assert descriptions == {'off': 'the intensity of the background is low',
                            'on': 'the intensity of the background is high'}

    with pytest.raises(Exception) as e:
        db_reader_empty.get_Name_and_Description_from_AnnotationTypeLabels("shape")
    assert str(e.value) == "no such table: AnnotationTypeLabels"


def test_get_Names_from_AnnotationTypeLabels(db_reader, db_reader_empty):
    names = db_reader._get_Names_from_AnnotationTypeLabels()
    assert names == ['c', 's', 'c1', 'c2', 'c3', 'on', 'off']

    with pytest.raises(Exception) as e:
        db_reader_empty._get_Names_from_AnnotationTypeLabels()
    assert str(e.value) == "no such table: AnnotationTypeLabels"


def test_get_Id_map_to_Names_from_AnnotationTypeLabels(db_reader, db_reader_empty):
    mapping = db_reader.get_Id_map_to_Names_from_AnnotationTypeLabels()
    assert mapping == {1: 'c', 2: 's', 3: 'c1', 4: 'c2', 5: 'c3', 6: 'on', 7: 'off'}

    with pytest.raises(Exception) as e:
        db_reader_empty.get_Id_map_to_Names_from_AnnotationTypeLabels()
    assert str(e.value) == "no such table: AnnotationTypeLabels"


def test_get_Id_from_AnnotationTypeLabels(db_reader, db_reader_empty):
    label_id = db_reader._get_Id_from_AnnotationTypeLabels(("shape", "c"))
    assert label_id == 1

    with pytest.raises(Exception) as e:
        db_reader._get_Id_from_AnnotationTypeLabels(("light", "c1"))
    assert str(e.value) == "Could not find a label from group light with name c1. Are you sure it's been added into " \
                           "the database? "

    with pytest.raises(Exception) as e:
        db_reader_empty.get_Id_map_to_Names_from_AnnotationTypeLabels()
    assert str(e.value) == "no such table: AnnotationTypeLabels"


def test_get_Ids_from_AnnotationTypeLabels(db_reader, db_reader_empty):
    assert db_reader._get_Ids_from_AnnotationTypeLabels("light") == [(7,), (6,)]

    with pytest.raises(Exception) as e:
        db_reader._get_Ids_from_AnnotationTypeLabels("test")
    assert str(e.value) == "Could not find labels from group ('test',) Are you sure it's been added into the database?"

    with pytest.raises(Exception) as e:
        db_reader_empty._get_Ids_from_AnnotationTypeLabels("light")
    assert str(e.value) == "no such table: AnnotationTypeLabels"


def test_get_AnnotationTypeLabelId_from_Annotations(db_reader, db_reader_empty):
    assert db_reader.get_AnnotationTypeLabelId_from_Annotations("shape") == \
           [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]

    with pytest.raises(Exception) as e:
        db_reader.get_AnnotationTypeLabelId_from_Annotations("test")
    assert str(e.value) == "Could not find labels from group ('test',) Are you sure it's been added into the database?"

    with pytest.raises(Exception) as e:
        db_reader_empty.get_AnnotationTypeLabelId_from_Annotations("light")
    assert str(e.value) == "no such table: Annotations"


def test_get_SliceInVolume_from_Volumes(db_reader, db_reader_empty):
    assert db_reader._get_SliceInVolume_from_Volumes([1, 2, 3, 4, 5, 10, 11, 12, 42]) == [0, 1, 2, 3, 4, 9, 0, 1, 1]

    with pytest.raises(Exception) as e:
        db_reader._get_SliceInVolume_from_Volumes([1, 43])
    assert str(e.value) == "Only 1 of 2 frames are in the database"

    with pytest.raises(Exception) as e:
        db_reader_empty._get_SliceInVolume_from_Volumes([1, 2, 3, 4, 5, 10, 11, 12, 42])
    assert str(e.value) == "no such table: Volumes"


def test__get_VolumeId_from_Volumes(db_reader, db_reader_empty):
    assert db_reader._get_VolumeId_from_Volumes([1, 2, 3, 11, 12, 13]) == [0, 0, 0, 1, 1, 1]
    assert db_reader._get_VolumeId_from_Volumes([11, 12, 13, 1, 2, 3]) == [0, 0, 0, 1, 1, 1]

    # does not preserve order
    assert db_reader._get_VolumeId_from_Volumes([11, 12, 13, 1, 2, 3]) != [1, 1, 1, 0, 0, 0]
    with pytest.raises(Exception) as e:
        db_reader_empty._get_VolumeId_from_Volumes([11, 12, 13, 1, 2, 3])
    assert str(e.value) == "no such table: Volumes"

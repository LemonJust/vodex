import pytest
from pathlib import Path
from sqlite3 import connect, Connection
from vodex import DbWriter, VolumeManager, Annotation, Labels, Cycle, Timeline

TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')
SPLIT_MOVIE_DIR = Path(TEST_DATA, "test_movie")
TEST_DB = Path(TEST_DATA, "test.db")

VOLUME_M = VolumeManager.from_dir(SPLIT_MOVIE_DIR, 10, fgf=0)
FILE_M = VOLUME_M.file_manager
FRAME_M = VOLUME_M.frame_manager
N_FRAMES = 42


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
def db_writer() -> DbWriter:
    return DbWriter.create()


@pytest.fixture
def db_writer_loaded() -> DbWriter:
    return DbWriter.load(TEST_DB)


def test_save(db_writer_loaded):
    test_db_file = "test_dbwriter.db"
    db_writer_loaded.save(test_db_file)
    # Assert that the file 'test_db.db' has been created and is not empty
    assert Path(test_db_file).is_file()
    assert Path(test_db_file).stat().st_size > 0
    # Clean up: delete the test file
    Path(test_db_file).unlink()


def test_create(db_writer):
    # Assert that the new db_writer object is created and its connection is not None
    assert db_writer is not None
    assert db_writer.connection is not None


def test_load(db_writer_loaded):
    # Assert that the loaded db_writer object is created and its connection is not None
    assert db_writer_loaded is not None
    assert db_writer_loaded.connection is not None


def test_populate(db_writer):
    db_writer.populate(VOLUME_M, [SHAPE_AN, CNUM_AN, LIGHT_AN])
    # Assert that the required tables are created and the populate methods are called
    cursor = db_writer.connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    assert ('Options',) in tables
    assert ('Files',) in tables
    assert ('Frames',) in tables
    assert ('Volumes',) in tables
    assert ('AnnotationTypes',) in tables
    assert ('AnnotationTypeLabels',) in tables
    assert ('Annotations',) in tables
    cursor.close()


def test_add_annotations(db_writer):
    # Create a list of Annotation objects to pass to the add_annotations method
    annotations = [CNUM_AN, LIGHT_AN]
    db_writer.populate(VOLUME_M, [SHAPE_AN])
    db_writer.add_annotations(annotations)
    # Assert that the required tables are created and the populate methods are called
    cursor = db_writer.connection.cursor()
    cursor.execute("SELECT * FROM Annotations")
    result = cursor.fetchall()
    assert len(result) == 126
    assert (10, 2) in result
    assert (42, 1) in result
    assert (3, 3) in result
    assert (12, 4) in result
    assert (29, 5) in result
    assert (27, 6) in result
    assert (42, 7) in result
    cursor.close()


def test_get_n_frames(db_writer_loaded):
    # Add some frames to the database
    assert db_writer_loaded._get_n_frames() == 42


def test_create_tables(db_writer):
    db_writer._create_tables()
    cursor = db_writer.connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    assert ('Options',) in tables
    assert ('Files',) in tables
    assert ('Frames',) in tables
    assert ('Volumes',) in tables
    assert ('AnnotationTypes',) in tables
    assert ('AnnotationTypeLabels',) in tables
    assert ('Annotations',) in tables
    assert ('Cycles',) in tables
    assert ('CycleIterations',) in tables
    cursor.close()


def test_populate_Options(db_writer):
    db_writer._create_tables()
    db_writer._populate_Options(FILE_M, VOLUME_M)
    cursor = db_writer.connection.cursor()
    cursor.execute("SELECT * FROM Options")
    result = cursor.fetchall()
    assert len(result) == 5
    assert ("data_dir", SPLIT_MOVIE_DIR.as_posix(), None) in result
    assert ("frames_per_volume", '10', None) in result
    assert ("num_head_frames", '0', None) in result
    assert ("num_tail_frames", '2', None) in result
    assert ("num_full_volumes", '4', None) in result
    cursor.close()
    with pytest.raises(Exception) as e:
        db_writer._populate_Options(FILE_M, VOLUME_M)
    assert str(e.value) == "UNIQUE constraint failed: Options.Key"


def test_populate_Files(db_writer):
    db_writer._create_tables()
    db_writer._populate_Files(FILE_M)
    cursor = db_writer.connection.cursor()
    cursor.execute("SELECT * FROM Files")
    result = cursor.fetchall()
    assert len(result) == 3
    assert (1, 'mov0.tif', 7) in result
    assert (2, 'mov1.tif', 18) in result
    assert (3, 'mov2.tif', 17) in result
    cursor.close()
    with pytest.raises(Exception) as e:
        db_writer._populate_Files(FILE_M)
    assert str(e.value) == "UNIQUE constraint failed: Files.FileName"


def test_populate_Frames(db_writer):
    db_writer._create_tables()
    # Frames have FOREIGN KEY constraint to files:
    with pytest.raises(Exception) as e:
        db_writer._populate_Frames(FRAME_M)
    assert str(e.value) == "FOREIGN KEY constraint failed"
    db_writer._populate_Files(FILE_M)
    db_writer._populate_Frames(FRAME_M)
    cursor = db_writer.connection.cursor()
    cursor.execute("SELECT * FROM Frames")
    result = cursor.fetchall()
    assert len(result) == 42
    assert (1, 0, 1) in result
    assert (2, 1, 1) in result

    assert (8, 0, 2) in result
    assert (9, 1, 2) in result
    assert (10, 2, 2) in result

    assert (40, 14, 3) in result
    assert (41, 15, 3) in result
    assert (42, 16, 3) in result
    cursor.close()

    with pytest.raises(Exception) as e:
        db_writer._populate_Frames(FRAME_M)
    assert str(e.value) == "UNIQUE constraint failed: Frames.FrameInFile, Frames.FileId"


def test_populate_Volumes(db_writer):
    db_writer._create_tables()
    # Volumes have FOREIGN KEY constraint to frames:
    with pytest.raises(Exception) as e:
        db_writer._populate_Volumes(VOLUME_M)
    assert str(e.value) == "FOREIGN KEY constraint failed"
    db_writer._populate_Files(FILE_M)
    db_writer._populate_Frames(FRAME_M)
    db_writer._populate_Volumes(VOLUME_M)
    cursor = db_writer.connection.cursor()
    cursor.execute("SELECT * FROM Volumes")
    result = cursor.fetchall()
    assert len(result) == 42
    assert (1, 0, 0) in result
    assert (2, 0, 1) in result
    assert (17, 1, 6) in result
    assert (33, 3, 2) in result

    assert (42, -2, 1) in result
    cursor.close()

    with pytest.raises(Exception) as e:
        db_writer._populate_Volumes(VOLUME_M)
    assert str(e.value) == "UNIQUE constraint failed: Volumes.VolumeId, Volumes.SliceInVolume"


def test_populate_AnnotationTypes(db_writer):
    db_writer._create_tables()
    db_writer._populate_AnnotationTypes(SHAPE_AN)
    cursor = db_writer.connection.cursor()
    cursor.execute("SELECT * FROM AnnotationTypes")
    result = cursor.fetchall()
    assert len(result) == 1
    assert (1, "shape", None) in result
    # add one more
    db_writer._populate_AnnotationTypes(CNUM_AN)
    cursor.execute("SELECT * FROM AnnotationTypes")
    result = cursor.fetchall()
    assert len(result) == 2
    assert (1, "shape", None) in result
    assert (2, "c label", None) in result
    cursor.close()

    with pytest.raises(Exception) as e:
        db_writer._populate_AnnotationTypes(SHAPE_AN)
    assert str(e.value) == "UNIQUE constraint failed: AnnotationTypes.Name"


def test_populate_AnnotationTypeLabels(db_writer):
    db_writer._create_tables()
    db_writer._populate_AnnotationTypes(SHAPE_AN)
    db_writer._populate_AnnotationTypeLabels(SHAPE_AN)

    cursor = db_writer.connection.cursor()

    cursor.execute("SELECT * FROM AnnotationTypeLabels")
    result = cursor.fetchall()
    assert len(result) == 2
    assert (1, 1, "c", "circle on the screen") in result
    assert (2, 1, "s", "square on the screen") in result

    # add more
    with pytest.raises(Exception) as e:
        db_writer._populate_AnnotationTypeLabels(CNUM_AN)
    assert str(e.value) == "NOT NULL constraint failed: AnnotationTypeLabels.AnnotationTypeId"
    db_writer._populate_AnnotationTypes(CNUM_AN)
    db_writer._populate_AnnotationTypeLabels(CNUM_AN)

    cursor.execute("SELECT * FROM AnnotationTypeLabels")
    result = cursor.fetchall()
    assert len(result) == 5
    assert (1, 1, "c", "circle on the screen") in result
    assert (2, 1, "s", "square on the screen") in result
    assert (3, 2, "c1", "written c1") in result
    assert (4, 2, "c2", "written c2") in result
    assert (5, 2, "c3", None) in result
    cursor.close()

    with pytest.raises(Exception) as e:
        db_writer._populate_AnnotationTypeLabels(SHAPE_AN)
    assert str(e.value) == "UNIQUE constraint failed: AnnotationTypeLabels.AnnotationTypeId, AnnotationTypeLabels.Name"


def test_populate_Annotations(db_writer):
    db_writer._create_tables()
    db_writer._populate_AnnotationTypes(SHAPE_AN)
    db_writer._populate_AnnotationTypeLabels(SHAPE_AN)
    # need Frames to get frame number
    with pytest.raises(Exception) as e:
        db_writer._populate_Annotations(SHAPE_AN)
    assert str(e.value) == "Number of frames in the annotation, 42,doesn't match the expected number of frames 0"
    db_writer._populate_Files(FILE_M)
    db_writer._populate_Frames(FRAME_M)
    db_writer._populate_Annotations(SHAPE_AN)

    cursor = db_writer.connection.cursor()
    cursor.execute("SELECT * FROM Annotations")
    result = cursor.fetchall()
    assert len(result) == 42
    assert (10, 2) in result
    assert (42, 1) in result

    # add more
    with pytest.raises(Exception) as e:
        db_writer._populate_Annotations(CNUM_AN)
    assert str(e.value) == "NOT NULL constraint failed: Annotations.AnnotationTypeLabelId"
    db_writer._populate_AnnotationTypes(CNUM_AN)
    db_writer._populate_AnnotationTypeLabels(CNUM_AN)
    db_writer._populate_Annotations(CNUM_AN)

    cursor.execute("SELECT * FROM Annotations")
    result = cursor.fetchall()
    assert len(result) == 84
    assert (10, 2) in result
    assert (42, 1) in result
    assert (3, 3) in result
    assert (12, 4) in result
    assert (29, 5) in result
    cursor.close()

    with pytest.raises(Exception) as e:
        db_writer._populate_Annotations(SHAPE_AN)
    assert str(e.value) == "UNIQUE constraint failed: Annotations.FrameId, Annotations.AnnotationTypeLabelId"


def test_populate_Cycles(db_writer):
    db_writer._create_tables()
    # need AnnotationTypes:
    with pytest.raises(Exception) as e:
        db_writer._populate_Cycles(SHAPE_AN)
    assert str(e.value) == "NOT NULL constraint failed: Cycles.AnnotationTypeId"
    db_writer._populate_AnnotationTypes(SHAPE_AN)
    db_writer._populate_Cycles(SHAPE_AN)

    cursor = db_writer.connection.cursor()
    cursor.execute("SELECT * FROM Cycles")
    result = cursor.fetchall()
    assert len(result) == 1
    cycle_json_string1 = '{"timing": [5, 10, 5], "label_order": [{"name": "c", "group": "shape", "description": "circle on the screen"}, {"name": "s", "group": "shape", "description": "square on the screen"}, {"name": "c", "group": "shape", "description": "circle on the screen"}]}'
    assert (1, 1, cycle_json_string1) in result

    # try adding another Cycle
    db_writer._populate_AnnotationTypes(CNUM_AN)
    db_writer._populate_Cycles(CNUM_AN)
    cursor.execute("SELECT * FROM Cycles")
    result = cursor.fetchall()
    assert len(result) == 2
    cycle_json_string2 = '{"timing": [10, 10, 10], "label_order": [{"name": "c1", "group": "c label", "description": "written c1"}, {"name": "c2", "group": "c label", "description": "written c2"}, {"name": "c3", "group": "c label"}]}'
    assert (1, 1, cycle_json_string1) in result
    assert (2, 2, cycle_json_string2) in result
    cursor.close()

    # try adding Timeline: LIGHT_AN
    db_writer._populate_AnnotationTypes(LIGHT_AN)
    with pytest.raises(Exception) as e:
        db_writer._populate_Cycles(LIGHT_AN)
    assert str(e.value) == "Annotation is not a Cycle"

    with pytest.raises(Exception) as e:
        db_writer._populate_Cycles(SHAPE_AN)
    assert str(e.value) == "UNIQUE constraint failed: Cycles.AnnotationTypeId"


def test_populate_CycleIterations(db_writer):
    db_writer._create_tables()

    # need Frames and Cycles
    with pytest.raises(Exception) as e:
        db_writer._populate_CycleIterations(SHAPE_AN)
    assert str(e.value) == "Number of frames in the annotation, 42,doesn't match the expected number of frames 0"

    db_writer._populate_Files(FILE_M)
    db_writer._populate_Frames(FRAME_M)
    with pytest.raises(Exception) as e:
        db_writer._populate_CycleIterations(SHAPE_AN)
    assert str(e.value) == "Fill out AnnotationTypes and Cycles first."

    db_writer._populate_AnnotationTypes(SHAPE_AN)
    with pytest.raises(Exception) as e:
        db_writer._populate_CycleIterations(SHAPE_AN)
    assert str(e.value) == "Fill out AnnotationTypes and Cycles first."

    db_writer._populate_Cycles(SHAPE_AN)
    db_writer._populate_CycleIterations(SHAPE_AN)

    cursor = db_writer.connection.cursor()
    cursor.execute("SELECT * FROM CycleIterations")
    result = cursor.fetchall()
    assert len(result) == 42
    assert (1, 1, 0) in result
    assert (42, 1, 2) in result

    # try adding another Cycle
    db_writer._populate_AnnotationTypes(CNUM_AN)
    db_writer._populate_Cycles(CNUM_AN)
    db_writer._populate_CycleIterations(CNUM_AN)
    cursor.execute("SELECT * FROM CycleIterations")
    result = cursor.fetchall()
    assert len(result) == 84
    assert (1, 1, 0) in result
    assert (42, 1, 2) in result
    assert (1, 2, 0) in result
    assert (42, 2, 1) in result
    cursor.close()

    # try adding again
    with pytest.raises(Exception) as e:
        db_writer._populate_CycleIterations(SHAPE_AN)
    assert str(e.value) == "UNIQUE constraint failed: CycleIterations.FrameId, CycleIterations.CycleId"

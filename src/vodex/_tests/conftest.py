import pytest
from vodex import *
from pathlib import Path
import tifffile as tif


def prepare_shape_cycle():
    shape = Labels("shape", ["c", "s"],
                   state_info={"c": "circle on the screen", "s": "square on the screen"})
    shape_cycle = Cycle([shape.c, shape.s, shape.c], [5, 10, 5])
    return shape, shape_cycle


def prepare_cnum_cycle():
    cnum = Labels("c label", ['c1', 'c2', 'c3'],
                  state_info={'c1': 'written c1', 'c2': 'written c2'})
    cnum_cycle = Cycle([cnum.c1, cnum.c2, cnum.c3], [10, 10, 10])
    return cnum, cnum_cycle


def prepare_light_tml():
    light = Labels("light", ["on", "off"],
                   group_info="Information about the light",
                   state_info={"on": "the intensity of the background is high",
                               "off": "the intensity of the background is low"})
    light_tml = Timeline([light.off, light.on, light.off], [10, 20, 12])
    return light, light_tml


# annotations to create an experiment
def prepare_annotations():
    shape, shape_cycle = prepare_shape_cycle()
    cnum, cnum_cycle = prepare_cnum_cycle()
    light, light_tml = prepare_light_tml()

    shape_an = Annotation.from_cycle(42, shape, shape_cycle)
    cnum_an = Annotation.from_cycle(42, cnum, cnum_cycle)
    light_an = Annotation.from_timeline(42, light, light_tml)

    return shape_an, cnum_an, light_an


#
TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')

# test data movie, where all the data is in one file
FULL_MOVIE_DIR = TEST_DATA
FULL_MOVIE_NAMES = ["test_movie.tif"]
FULL_MOVIE = Path(TEST_DATA, FULL_MOVIE_NAMES[0])
FULL_MOVIE_FRAMES = [42]

# test data movie, where all the data is split into 3 files
SPLIT_MOVIE_DIR = Path(TEST_DATA, "test_movie")
SPLIT_MOVIE_NAMES = ["mov0.tif", "mov1.tif", "mov2.tif"]
SPLIT_MOVIE = [Path(SPLIT_MOVIE_DIR, mov) for mov in SPLIT_MOVIE_NAMES]
SPLIT_MOVIE_FRAMES = [7, 18, 17]

# core classes
N_FRAMES = 42
VOLUME_M = VolumeManager.from_dir(SPLIT_MOVIE_DIR, 10, fgf=0)
FILE_M = VOLUME_M.file_manager
FRAME_M = VOLUME_M.frame_manager

SHAPE_CYCLE = prepare_shape_cycle()[1]
CNUM_CYCLE = prepare_cnum_cycle()[1]
LIGHT_TML = prepare_light_tml()[1]
SHAPE_AN, CNUM_AN, LIGHT_AN = prepare_annotations()

# supported extensions information
SUPPORTED_EXTENSIONS = ['tif', 'tiff']
# TIFF IMAGES __________________________________________________________________________________________________
# frame characteristics
FRAME_SIZE = (200, 200)
# saved images of volumes 0 and 1 and the last two frames (tail)
VOLUMES_TAIL = tif.imread(Path(TEST_DATA, 'loader_test', "volumes_tail.tif").as_posix())
# full volumes 0 and 1
VOLUMES_FRAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
VOLUMES_INDICES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
VOLUMES_0_1 = tif.imread(Path(TEST_DATA, 'loader_test', "volumes_1_2.tif").as_posix())
# top half slice of the volumes 0 and 1
SLICES_FRAMES = [0, 1, 2, 3, 4,
                 10, 11, 12, 13, 14]
SLICES_INDICES = [0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1]
HALF_VOLUMES_0_1 = tif.imread(Path(TEST_DATA, 'loader_test', "half_volumes_1_2.tif").as_posix())
# individual frames
FRAMES = [0, 1, 40, 41]
FRAMES_1_2_41_42 = tif.imread(Path(TEST_DATA, 'loader_test', "frames_1_2_41_42.tif").as_posix())

# for testing loading of the slices
VOLUMES_0_TAIL_SLICES_0_1 = tif.imread(Path(TEST_DATA, 'loader_test', "volumes_0_tail_slices_0_1.tif").as_posix())
SLICES_0_1 = tif.imread(Path(TEST_DATA, 'loader_test', "slices_0_1.tif").as_posix())
SLICES_0 = tif.imread(Path(TEST_DATA, 'loader_test', "slices_0.tif").as_posix())
SLICES_2 = tif.imread(Path(TEST_DATA, 'loader_test', "slices_2.tif").as_posix())
# reshape into 4D array
VOLUMES_0_TAIL_SLICES_0_1 = VOLUMES_0_TAIL_SLICES_0_1.reshape((2, 2, 200, 200))
SLICES_0_1 = SLICES_0_1.reshape((5, 2, 200, 200))
SLICES_0 = SLICES_0.reshape((5, 1, 200, 200))
SLICES_2 = SLICES_2.reshape((4, 1, 200, 200))


# before tests run, will create database with this name:
@pytest.fixture(autouse=True, scope='session')
def TEST_DB(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('tmp')
    test_db = Path(datadir, "test.db")
    vm = VolumeManager.from_dir(SPLIT_MOVIE_DIR, 10, fgf=0)
    experiment = Experiment.create(vm, [SHAPE_AN, CNUM_AN, LIGHT_AN])
    experiment.save(test_db)
    return test_db

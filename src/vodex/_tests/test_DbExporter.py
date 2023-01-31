import pytest
from pathlib import Path
from vodex import (DbExporter,
                   FileManager,
                   TimeLabel,
                   VolumeManager,
                   Annotation,
                   Labels,
                   Timeline,
                   Cycle)

TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')
# test data movie, where all the data is split into 3 files
SPLIT_MOVIE_DIR = Path(TEST_DATA, "test_movie")
SPLIT_MOVIE_NAMES = ["mov0.tif", "mov1.tif", "mov2.tif"]
SPLIT_MOVIE_FRAMES = [7, 18, 17]

TEST_DB = Path(TEST_DATA, "test.db")


@pytest.fixture
def db_exporter():
    return DbExporter.load(TEST_DB)


def test_load(db_exporter):
    assert isinstance(db_exporter, DbExporter)


def test_reconstruct_file_manager(db_exporter):
    fm = db_exporter.reconstruct_file_manager()
    assert isinstance(fm, FileManager)
    assert fm.data_dir == SPLIT_MOVIE_DIR
    assert fm.file_names == SPLIT_MOVIE_NAMES
    assert fm.num_frames == SPLIT_MOVIE_FRAMES


def test_reconstruct_volume_manager(db_exporter):
    vm = db_exporter.reconstruct_volume_manager()
    assert isinstance(vm, VolumeManager)
    assert vm.fpv == 10
    assert vm.n_head == 0
    assert vm.n_tail == 2


def test_reconstruct_labels(db_exporter):
    labels = db_exporter.reconstruct_labels("shape")
    assert isinstance(labels, Labels)
    assert labels.state_names == ['c', 's']


def test_reconstruct_timeline(db_exporter):
    timeline = db_exporter.reconstruct_timeline("light")
    assert isinstance(timeline, Timeline)
    assert timeline.duration == [10, 20, 12]
    assert timeline.label_order == [TimeLabel("off", group="light"),
                                    TimeLabel("on", group="light"),
                                    TimeLabel("off", group="light")]


def test_reconstruct_cycle(db_exporter):
    assert db_exporter.reconstruct_cycle("light") is None
    cycle = db_exporter.reconstruct_cycle("shape")
    assert isinstance(cycle, Cycle)
    assert cycle.label_order == [TimeLabel("c", group="shape"),
                                 TimeLabel("s", group="shape"),
                                 TimeLabel("c", group="shape")]
    assert cycle.duration == [5, 10, 5]


def test_reconstruct_annotations(db_exporter):
    annotations = db_exporter.reconstruct_annotations()
    assert len(annotations) == 3
    for annotation in annotations:
        assert isinstance(annotation, Annotation)
        assert annotation.n_frames == 42

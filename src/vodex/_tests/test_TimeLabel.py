import json
from pathlib import Path
import pytest
from vodex import TimeLabel, Labels


@pytest.fixture
def time_label():
    return TimeLabel("on", "The light is on", "light")



def test_time_label_init(time_label):
    assert time_label.name == "on"
    assert time_label.description == "The light is on"
    assert time_label.group == "light"


def test_time_label_str(time_label):
    assert str(time_label) == "on : The light is on. Group: light"


def test_time_label_repr(time_label):
    assert repr(time_label) == "on : The light is on. Group: light"


def test_time_label_eq(time_label):
    assert time_label == TimeLabel("on", "The light is on", "light")
    assert time_label == TimeLabel("on", group="light")
    assert time_label != TimeLabel("on")
    assert time_label != TimeLabel("on", "The light is on")
    assert time_label != TimeLabel("off", "The light is off", "light")
    assert time_label != TimeLabel("on", "The light is on", "different group")
    assert time_label.__eq__("not a time label") == NotImplemented


def test_time_label_to_dict(time_label):
    assert time_label.to_dict() == {'name': 'on', 'description': 'The light is on', 'group': 'light'}
    assert TimeLabel("on").to_dict() == {'name': 'on'}
    assert TimeLabel("on", description='The light is on').to_dict() == {'name': 'on', 'description': 'The light is on'}
    assert TimeLabel("on", group='light').to_dict() == {'name': 'on', 'group': 'light'}


def test_time_label_from_dict():
    time_label = TimeLabel.from_dict({'name': 'on', 'description': 'The light is on', 'group': 'light'})
    assert time_label == TimeLabel("on", "The light is on", "light")
    time_label = TimeLabel.from_dict({'name': 'on'})
    assert time_label == TimeLabel("on")
    time_label = TimeLabel.from_dict({'name': 'on', 'description': 'The light is on'})
    assert time_label == TimeLabel("on", description="The light is on")
    time_label = TimeLabel.from_dict({'name': 'on', 'group': 'light'})
    assert time_label == TimeLabel("on", group="light")


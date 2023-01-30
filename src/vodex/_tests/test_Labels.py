import pytest
from vodex import TimeLabel, Labels


@pytest.fixture
def labels():
    state_names = ["state1", "state2", "state3"]
    state_info = {"state1": "description of state1", "state2": "description of state2"}
    return Labels("group1", state_names, group_info="group1 description", state_info=state_info)


def test_init(labels):
    assert labels.group == "group1"
    assert labels.group_info == "group1 description"
    assert labels.state_names == ["state1", "state2", "state3"]
    assert isinstance(labels.states[0], TimeLabel)
    assert labels.states[0].name == "state1"
    assert labels.states[0].description == "description of state1"
    assert labels.states[0].group == "group1"
    assert labels.states[1].name == "state2"
    assert labels.states[1].description == "description of state2"
    assert labels.states[1].group == "group1"
    assert labels.states[2].name == "state3"
    assert labels.states[2].description is None
    assert labels.states[2].group == "group1"

    assert labels.state1 == labels.states[0]
    assert labels.state2 == labels.states[1]
    assert labels.state3 == labels.states[2]


def test_eq(labels):
    state_names = ["state1", "state2", "state3"]
    state_info = {"state1": "description of state1", "state2": "description of state2"}
    labels2 = Labels("group1", state_names, group_info="group1 description", state_info=state_info)
    assert labels == labels2
    assert labels.__eq__("label") == NotImplemented


def test_str(labels):
    expected_str = ('Label group : group1\n'
                    'States:\n'
                    'state1 : description of state1. Group: group1\n'
                    'state2 : description of state2. Group: group1\n'
                    'state3. Group: group1\n')
    assert str(labels) == expected_str


def test_repr(labels):
    expected_str = ('Label group : group1\n'
                    'States:\n'
                    'state1 : description of state1. Group: group1\n'
                    'state2 : description of state2. Group: group1\n'
                    'state3. Group: group1\n')
    assert repr(labels) == expected_str

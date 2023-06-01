import pytest
from vodex import TimeLabel, Labels
import pandas as pd


@pytest.fixture
def labels():
    state_names = ["state1", "state2", "state3"]
    state_info = {"state1": "description of state1", "state2": "description of state2"}
    return Labels("group1", state_names, group_info="group1 description", state_info=state_info)


@pytest.fixture
def labels_no_group_info():
    state_names = ["state1", "state2", "state3"]
    state_info = {"state1": "description of state1", "state2": "description of state2"}
    return Labels("group1", state_names, state_info=state_info)


@pytest.fixture
def labels_no_info():
    state_names = ["state1", "state2", "state3"]
    return Labels("group1", state_names)


def test_init(labels, labels_no_info):
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

    # now no info:
    assert labels_no_info.group == "group1"
    assert labels_no_info.group_info is None
    assert labels_no_info.state_names == ["state1", "state2", "state3"]
    assert isinstance(labels_no_info.states[0], TimeLabel)
    assert labels_no_info.states[0].name == "state1"
    assert labels_no_info.states[0].description is None
    assert labels_no_info.states[0].group == "group1"
    assert labels_no_info.states[1].name == "state2"
    assert labels_no_info.states[1].description is None
    assert labels_no_info.states[1].group == "group1"
    assert labels_no_info.states[2].name == "state3"
    assert labels_no_info.states[2].description is None
    assert labels_no_info.states[2].group == "group1"

    assert labels_no_info.state1 == labels.states[0]
    assert labels_no_info.state2 == labels.states[1]
    assert labels_no_info.state3 == labels.states[2]


def test_eq(labels):
    state_names = ["state1", "state2", "state3"]
    state_info = {"state1": "description of state1", "state2": "description of state2"}
    labels2 = Labels("group1", state_names, group_info="group1 description", state_info=state_info)
    # different order of states
    state_names = ["state2", "state1", "state3"]
    state_info = {"state1": "description of state1", "state2": "description of state2"}
    labels3 = Labels("group1", state_names, group_info="group1 description", state_info=state_info)

    assert labels == labels2
    assert labels == labels3
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


def test_from_dict(labels):
    d = {"state_names": ["state1", "state2", "state3"],
         "state_info": {"state1": "description of state1",
                        "state2": "description of state2"},
         "group": "group1",
         "group_info": "group1 description"}
    labels_from_dict = Labels.from_dict(d)
    assert labels_from_dict == labels


def test_from_df(labels, labels_no_group_info, labels_no_info):
    df_info = pd.DataFrame({"name": ["state1", "state2", "state3"],
                            "description": ["description of state1", "description of state2", None],
                            "group": ["group1", "group1", "group1"]})

    df_no_info = pd.DataFrame({"name": ["state1", "state2", "state3"],
                               "group": ["group1", "group1", "group1"]})

    labels_from_df = Labels.from_df(df_info)
    assert labels_from_df == labels_no_group_info
    assert labels_from_df != labels  # because of the group info

    labels_from_df = Labels.from_df(df_no_info)
    assert labels_from_df == labels_no_info
    assert labels_from_df == labels_no_group_info  # because descriptions are not checked for equality

    labels_from_df = Labels.from_df(df_no_info, group="group1")
    assert labels_from_df == labels_no_info
    assert labels_from_df == labels_no_group_info  # because descriptions are not checked for equality

    with pytest.raises(ValueError) as e:
        Labels.from_df(df_info, group="group2")
    assert str(e.value) == 'Group group2 not found in the dataframe.'

    df_no_info_2g = pd.DataFrame({"name": ["state1", "state2", "state3", "state21", "state22", "state23"],
                                  "group": ["group1", "group1", "group1", "group2", "group2", "group2"]})
    labels_from_df = Labels.from_df(df_no_info_2g, group="group1")
    assert labels_from_df == labels_no_info
    assert labels_from_df == labels_no_group_info

    with pytest.raises(ValueError) as e:
        Labels.from_df(df_no_info_2g)
    assert str(e.value) == "More than one group found in the dataframe: ['group1' 'group2']"

    df_info_multi = pd.DataFrame({"name": ["state1", "state2", "state3", "state1", "state2", "state3"],
                                  "description": ["description of state1", "description of state2", None,
                                                  "description of state1", "description of state2", None],
                                  "group": ["group1"] * 6})
    labels_from_df = Labels.from_df(df_info_multi)
    assert labels_from_df == labels_no_group_info
    assert labels_from_df != labels  # because of the group info

    df_info_multi = pd.DataFrame({"name": ["state1", "state2", "state3", "state1", "state2", "state3"],
                                  "description": ["description of state1", "description of state2", '',
                                                  '', '', ''],
                                  "group": ["group1"] * 6})
    labels_from_df = Labels.from_df(df_info_multi)
    assert labels_from_df == labels_no_group_info
    assert labels_from_df != labels  # because of the group info

    df_info_multi = pd.DataFrame({"name": ["state1", "state2", "state3", "state1", "state2", "state3"],
                                  "description": ["description of state1", "description of state2", None,
                                                  '', None, None],
                                  "group": ["group1"] * 6})
    labels_from_df = Labels.from_df(df_info_multi)
    assert labels_from_df == labels_no_group_info
    assert labels_from_df != labels  # because of the group info

    df_info_multi = pd.DataFrame({"name": ["state1", "state2", "state3", "state1", "state2", "state3"],
                                  "description": ["description of state1", "description of state2", None,
                                                  "description of state1", None, None],
                                  "group": ["group1"] * 6})
    labels_from_df = Labels.from_df(df_info_multi)
    assert labels_from_df == labels_no_group_info
    assert labels_from_df != labels  # because of the group info

    df_info_multi = pd.DataFrame({"name": ["state1", "state2", "state3", "state1", "state2", "state3"],
                                  "description": ["description of state1", "description of state2", None,
                                                  "another description of state1", None, None],
                                  "group": ["group1"] * 6})
    with pytest.raises(ValueError) as e:
        Labels.from_df(df_info_multi)
    assert "More than one description found for state state1" in str(e.value)

    df_info_multi = pd.DataFrame({"name": ["state1", "state2", "state3",
                                           "state1", "state2", "state3",
                                           "state1", "state2", "state3"],
                                  "description": ["description of state1", "description of state2", None,
                                                  "another description of state1", None, None,
                                                  "and another description of state1", None, None],
                                  "group": ["group1"] * 9})
    with pytest.raises(ValueError) as e:
        Labels.from_df(df_info_multi)
    assert "More than one description found for state state1" in str(e.value)

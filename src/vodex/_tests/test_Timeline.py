"""
Tests for the `vodex.Timeline` module.
"""

import pytest
from vodex import TimeLabel, Timeline
import numpy as np
import pandas as pd
import json


@pytest.fixture
def label_order():
    label1 = TimeLabel("name1", group="group1")
    label2 = TimeLabel("name2", group="group1")

    return [label1, label2, label1]


@pytest.fixture
def duration():
    return [1, 2, 3]


@pytest.fixture
def timeline(label_order, duration):
    return Timeline(label_order, duration)


def test_init(label_order, duration):
    # correct with List[int]
    timeline = Timeline(label_order, duration)
    assert timeline.name == label_order[0].group
    assert timeline.label_order == label_order
    assert timeline.duration == duration
    assert timeline.full_length == sum(duration)
    assert timeline.per_frame_list == [label_order[0], label_order[1], label_order[1],
                                       label_order[2], label_order[2], label_order[2]]
    # correct with npt.NDArray[int]
    timeline = Timeline(label_order, np.array([1, 2, 3]))
    assert timeline.name == label_order[0].group
    assert timeline.label_order == label_order
    assert timeline.duration == duration
    assert timeline.full_length == sum(duration)
    assert timeline.per_frame_list == [label_order[0], label_order[1], label_order[1],
                                       label_order[2], label_order[2], label_order[2]]
    # wrong with npt.NDArray[float]
    with pytest.raises(AssertionError) as e:
        Timeline(label_order, np.array([1, 2, 3.0]))
    assert str(e.value) == "timing should be a list of int"

    # wrong labels
    label_g2 = TimeLabel("name1", group="group2")
    with pytest.raises(AssertionError) as e:
        Timeline([label_order[0], label_order[1], label_g2], duration)
    assert str(e.value) == "All labels should be from the same group, but got group2 and group1"

    label_None1 = TimeLabel("name1")
    with pytest.raises(AssertionError) as e:
        Timeline([label_order[0], label_order[1], label_None1], duration)
    assert str(e.value) == "All labels should be from the same group, but got None and group1"

    label_None2 = TimeLabel("name2")
    label_None3 = TimeLabel("name3")
    with pytest.raises(AssertionError) as e:
        Timeline([label_None1, label_None2, label_None3], duration)
    assert str(e.value) == "All labels should be from the same group, label group can not be None"


def test_eq(label_order, duration):
    timeline1 = Timeline(label_order, duration)
    timeline2 = Timeline(label_order, duration)
    assert timeline1 == timeline2
    assert timeline2 == timeline1
    # different duration
    timeline3 = Timeline(label_order, [3, 2, 1])
    assert timeline1 != timeline3
    assert timeline3 != timeline1
    # different group
    label1 = TimeLabel("name1", group="group2")
    label2 = TimeLabel("name2", group="group2")
    timeline4 = Timeline([label1, label2, label1], duration)
    assert timeline1 != timeline4
    assert timeline4 != timeline1
    # different names
    label1 = TimeLabel("name1", group="group1")
    label2 = TimeLabel("name3", group="group1")
    timeline5 = Timeline([label1, label2, label1], duration)
    assert timeline1 != timeline5
    assert timeline5 != timeline1

    assert timeline1.__eq__("Timeline") == NotImplemented


def test_get_label_per_frame(timeline, label_order):
    assert timeline.get_label_per_frame() == [label_order[0], label_order[1], label_order[1],
                                              label_order[2], label_order[2], label_order[2]]


def test_str(timeline):
    assert str(timeline) == ('Timeline : group1\n'
                             'Length: 6\n'
                             'Label name1: for 1 frames\n'
                             'Label name2: for 2 frames\n'
                             'Label name1: for 3 frames\n')


def test_repr(timeline):
    assert repr(timeline) == ('Timeline : group1\n'
                              'Length: 6\n'
                              'Label name1: for 1 frames\n'
                              'Label name2: for 2 frames\n'
                              'Label name1: for 3 frames\n')


def test_from_dict(timeline, label_order, duration):
    d = {'timing': duration,
         'label_order': [{'name': label.name,
                          'group': label.group,
                          'description': label.description} for label in label_order]
         }
    assert Timeline.from_dict(d) == timeline


def test_from_json(timeline, label_order, duration):
    j = json.dumps({'timing': duration,
                    'label_order': [{'name': label.name,
                                     'group': label.group,
                                     'description': label.description} for label in label_order]})
    assert Timeline.from_json(j) == timeline


def test_from_df(timeline, label_order, duration):
    df1 = pd.DataFrame({'duration_frames': duration,
                        'name': [label.name for label in label_order],
                        'group': [label.group for label in label_order],
                        'description': [label.description for label in label_order]})

    df1_timeline = Timeline.from_df(df1)
    assert df1_timeline == timeline

    assert df1_timeline.label_order[0].description == label_order[0].description
    assert df1_timeline.label_order[1].description == label_order[1].description
    assert df1_timeline.label_order[2].description == label_order[2].description

    df2 = pd.DataFrame({'duration_volumes': duration,
                        'name': [label.name for label in label_order],
                        'group': [label.group for label in label_order],
                        'description': [label.description for label in label_order]})

    with pytest.raises(AssertionError) as e:
        Timeline.from_df(df2)
    assert str(e.value) == 'if duration_frames is not in the dataframe, timing_conversion must be provided'

    with pytest.raises(AssertionError) as e:
        Timeline.from_df(df2, timing_conversion=1)
    assert str(e.value) == "timing_conversion must be a dictionary"

    with pytest.raises(AssertionError) as e:
        Timeline.from_df(df2, timing_conversion={'volumes': 1})
    assert str(e.value) == "frames must be in the timing_conversion dictionary"

    with pytest.raises(AssertionError) as e:
        Timeline.from_df(df2, timing_conversion={'frames': 1})
    assert str(e.value) == "timing_conversion dictionary must have at " \
                           "least one of the following keys in addition to 'frames': volumes"

    df2_timeline = Timeline.from_df(df2, timing_conversion={'frames': 1, 'volumes': 1})
    assert df2_timeline == timeline

    with pytest.raises(AssertionError) as e:
        Timeline.from_df(df2, timing_conversion={'frames': 1, 'volumes': 2})
    assert str(e.value) == 'duration in frames must be integer after conversion from volumes'


def test_to_dict(timeline):
    assert timeline.to_dict() == {'timing': [1, 2, 3],
                                  'label_order': [{'name': 'name1', 'group': 'group1'},
                                                  {'name': 'name2', 'group': 'group1'},
                                                  {'name': 'name1', 'group': 'group1'}]}


def test_to_json(timeline):
    assert timeline.to_json() == json.dumps({'timing': [1, 2, 3],
                                             'label_order': [{'name': 'name1', 'group': 'group1'},
                                                             {'name': 'name2', 'group': 'group1'},
                                                             {'name': 'name1', 'group': 'group1'}]})


def test_to_df(timeline):
    df = pd.DataFrame({'duration_frames': [1, 2, 3],
                       'name': ['name1', 'name2', 'name1'],
                       'group': ['group1', 'group1', 'group1'],
                       'description': [None, None, None]})
    pd.testing.assert_frame_equal(timeline.to_df(), df, check_dtype=False)


def test_to_df_timing_conversion(timeline):
    df = pd.DataFrame({'duration_frames': [1, 2, 3],
                       'duration_volumes': [10, 20, 30],
                       'duration_seconds': [0.1, 0.2, 0.3],
                       'name': ['name1', 'name2', 'name1'],
                       'group': ['group1', 'group1', 'group1'],
                       'description': [None, None, None]})
    pd.testing.assert_frame_equal(timeline.to_df(timing_conversion={'frames': 10, 'volumes': 100, 'seconds': 1}),
                                  df, check_dtype=False)

    with pytest.raises(AssertionError) as e:
        timeline.to_df(timing_conversion={'seconds': 10, 'volumes': 100})
    assert str(e.value) == "frames must be in the timing_conversion dictionary"

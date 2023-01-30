"""
Tests for the `vodex.Timeline` module.
"""

import pytest
from typing import List
from vodex import TimeLabel, Timeline
import numpy as np


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

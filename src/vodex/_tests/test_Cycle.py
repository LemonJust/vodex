"""
Tests for the `vodex.core` module.
"""
import json
from pathlib import Path
import pytest
from vodex import TimeLabel, Labels, Cycle

TEST_DATA = Path(Path(__file__).parent.resolve(), 'data')

import pytest
from typing import List, Union
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
def cycle(label_order, duration):
    return Cycle(label_order, duration)


def test_cycle_init(label_order, duration):
    # correct with List[int]
    cycle = Cycle(label_order, duration)
    assert cycle.name == label_order[0].group
    assert cycle.label_order == label_order
    assert cycle.duration == duration
    assert cycle.cycle_length == sum(duration)
    assert cycle.per_frame_list == [label_order[0], label_order[1], label_order[1],
                                    label_order[2], label_order[2], label_order[2]]
    # correct with npt.NDArray[int]
    cycle = Cycle(label_order, np.array([1, 2, 3]))
    assert cycle.name == label_order[0].group
    assert cycle.label_order == label_order
    assert cycle.duration == duration
    assert cycle.cycle_length == sum(duration)
    assert cycle.per_frame_list == [label_order[0], label_order[1], label_order[1],
                                    label_order[2], label_order[2], label_order[2]]
    # wrong with npt.NDArray[float]
    with pytest.raises(AssertionError) as e:
        Cycle(label_order, np.array([1, 2, 3.0]))
    assert str(e.value) == "timing should be a list of int"

    # wrong labels
    label_g2 = TimeLabel("name1", group="group2")
    with pytest.raises(AssertionError) as e:
        Cycle([label_order[0], label_order[1], label_g2], duration)
    assert str(e.value) == "All labels should be from the same group, but got group2 and group1"

    label_None1 = TimeLabel("name1")
    with pytest.raises(AssertionError) as e:
        Cycle([label_order[0], label_order[1], label_None1], duration)
    assert str(e.value) == "All labels should be from the same group, but got None and group1"

    label_None2 = TimeLabel("name2")
    label_None3 = TimeLabel("name3")
    with pytest.raises(AssertionError) as e:
        Cycle([label_None1, label_None2, label_None3], duration)
    assert str(e.value) == "All labels should be from the same group, label group can not be None"


def test_cycle_eq(label_order, duration):
    cycle1 = Cycle(label_order, duration)
    cycle2 = Cycle(label_order, duration)
    assert cycle1 == cycle2
    assert cycle2 == cycle1
    # different duration
    cycle3 = Cycle(label_order, [3, 2, 1])
    assert cycle1 != cycle3
    assert cycle3 != cycle1
    # different group
    label1 = TimeLabel("name1", group="group2")
    label2 = TimeLabel("name2", group="group2")
    cycle4 = Cycle([label1, label2, label1], duration)
    assert cycle1 != cycle4
    assert cycle4 != cycle1
    # different names
    label1 = TimeLabel("name1", group="group1")
    label2 = TimeLabel("name3", group="group1")
    cycle5 = Cycle([label1, label2, label1], duration)
    assert cycle1 != cycle5
    assert cycle5 != cycle1

    assert cycle1.__eq__("Cycle") == NotImplemented


def test_cycle_str(cycle):
    assert str(cycle) == ('Cycle : group1\n'
                          'Length: 6\n'
                          'Label name1: for 1 frames\n'
                          'Label name2: for 2 frames\n'
                          'Label name1: for 3 frames\n')


def test_cycle_repr(cycle):
    assert repr(cycle) == ('Cycle : group1\n'
                           'Length: 6\n'
                           'Label name1: for 1 frames\n'
                           'Label name2: for 2 frames\n'
                           'Label name1: for 3 frames\n')


def test_cycle_fit_frames(cycle):
    assert cycle.fit_frames(0) == 0
    assert cycle.fit_frames(5) == 1
    assert cycle.fit_frames(6) == 1
    assert cycle.fit_frames(7) == 2
    with pytest.raises(AssertionError) as e:
        cycle.fit_frames(-2)
    assert str(e.value) == "n_frames must be positive"


def test_cycle_fit_labels_to_frames(cycle, label_order):
    assert cycle.fit_labels_to_frames(0) == []
    assert cycle.fit_labels_to_frames(5) == [label_order[0], label_order[1], label_order[1],
                                             label_order[2], label_order[2]]
    assert cycle.fit_labels_to_frames(6) == [label_order[0], label_order[1], label_order[1],
                                             label_order[2], label_order[2], label_order[2]]
    assert cycle.fit_labels_to_frames(7) == [label_order[0], label_order[1], label_order[1],
                                             label_order[2], label_order[2], label_order[2], label_order[0]]


def test_cycle_fit_cycles_to_frames(cycle):
    assert cycle.fit_cycles_to_frames(0) == []
    assert cycle.fit_cycles_to_frames(5) == [0, 0, 0, 0, 0]
    assert cycle.fit_cycles_to_frames(6) == [0, 0, 0, 0, 0, 0]
    assert cycle.fit_cycles_to_frames(7) == [0, 0, 0, 0, 0, 0, 1]


def test_cycle_to_dict(cycle):
    assert cycle.to_dict() == {'timing': [1, 2, 3],
                               'label_order': [{'name': 'name1', 'group': 'group1'},
                                               {'name': 'name2', 'group': 'group1'},
                                               {'name': 'name1', 'group': 'group1'}]}


def test_cycle_to_json(cycle):
    assert cycle.to_json() == json.dumps({'timing': [1, 2, 3],
                                          'label_order': [{'name': 'name1', 'group': 'group1'},
                                                          {'name': 'name2', 'group': 'group1'},
                                                          {'name': 'name1', 'group': 'group1'}]})


def test_cycle_from_dict(cycle):
    d = {'timing': [1, 2, 3],
         'label_order': [{'name': 'name1', 'group': 'group1'},
                                    {'name': 'name2', 'group': 'group1'},
                                    {'name': 'name1', 'group': 'group1'}]}
    assert Cycle.from_dict(d) == cycle


def test_cycle_from_json(cycle):
    j = json.dumps({'timing': [1, 2, 3],
                    'label_order': [{'name': 'name1', 'group': 'group1'},
                                    {'name': 'name2', 'group': 'group1'},
                                    {'name': 'name1', 'group': 'group1'}]})
    assert Cycle.from_json(j) == cycle

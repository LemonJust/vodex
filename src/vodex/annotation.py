"""
This module provides the classes for constructing time annotations for imaging data.

'TimeLabel', 'Labels', 'Cycle', 'Timeline', and 'Annotation' classes help to construct and store time annotations. The
'TimeLabel' class stores information about specific time-located events during the experiment, such as a specific
condition described by a group and label. The 'Labels' class stores information about a group of time labels,
such as temperature, light, sound, image on the screen, drug, or behavior. The 'Cycle' class stores and preprocesses
information about repeated cycles of labels, useful for creating annotations for periodic conditions. The 'Timeline'
class stores and preprocesses information about the sequence of labels, useful for creating annotations for
non-periodic conditions. Finally, the 'Annotation' class stores and preprocesses information about the time annotation
of the experiment; it uses either the 'Cycle' or 'Timeline' classes to initialize the annotation.
"""
import json
from itertools import groupby
from typing import Union, List, Optional, Tuple, Dict, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .utils import list_of_int


class TimeLabel:
    """
    Stores information about a particular time-located event during the experiment: any specific condition,
    described by a group and the label.
    For example: group 'light', label 'on': the light was on; group 'light', label 'off': the light was off.

    Args:
        name: the name for the time label. This is a unique identifier of the label.
                    Different labels must have different names.
                    Different labels are compared based on their names, so the same name means it is the same event.
        description: a detailed description of the label. This is to give you more info, but it is not used for
            anything else.
        group: the group that the label belongs to.

    Attributes:
        name: the name for the time label. This is a unique identifier of the label.
                    Different labels must have different names.
                    Different labels are compared based on their names, so the same name means it is the same event.
        description: a detailed description of the label. This is to give you more info, but it is not used for
            anything else.
        group: the group that the label belongs to.
    """

    def __init__(self, name: str, description: str = None, group: str = None):
        self.name: str = name
        self.group: str = group
        self.description: str = description

    def __str__(self):
        description = self.name
        if self.description is not None:
            description = description + " : " + self.description
        if self.group is not None:
            description = description + ". Group: " + self.group
        return description

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash((self.name, self.group))

    def __eq__(self, other):
        if isinstance(other, TimeLabel):
            # comparing by name
            same_name = self.name == other.name
            if self.group is not None or other.group is not None:
                same_group = self.group == other.group
                return same_name and same_group
            else:
                return same_name
        else:
            print(
                f"__eq__ is Not Implemented for {TimeLabel} and {type(other)}")
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_dict(self) -> dict:
        """
        Put all the information about a TimeLabel object into a dictionary.

        Returns:
            a dictionary with fields 'name', 'group', 'description' which store the corresponding attributes.
        """
        d = {'name': self.name}
        if self.group is not None:
            d['group'] = self.group
        if self.description is not None:
            d['description'] = self.description
        return d

    @classmethod
    def from_dict(cls, d: dict):
        """
        Create a TimeLabel object from a dictionary.

        Returns:
            (TimeLabel): a TimeLabel object with attributes 'name', 'group', 'description'
                    filled from the dictionary fields.
        """
        description = None
        group = None
        if 'description' in d:
            description = d['description']
        if 'group' in d:
            group = d['group']
        return cls(d['name'], description=description, group=group)


class Labels:
    """
    Stores information about a group of time labels. Any specific aspect of the experiment that you want to document.
        For example: temperature|light|sound|image on the screen|drug|behaviour ... etc.

    Args:
        group : the name of the group
        group_info : description of what this group is about. Just for storing the information.
        state_names: the state names
        state_info: description of each individual state {state name : description}. Just for storing the information.

    Attributes:
        group: the name of the group
        group_info: description of what this group is about. Just for storing the information.
        state_names: the state names
        states: list of states, each state as a TimeLabel object

    """

    def __init__(self, group: str, state_names: List[str],
                 group_info: str = None, state_info: Optional[dict] = None):

        if state_info is None:
            state_info = {}
        self.group = group
        self.group_info = group_info
        self.state_names = state_names
        # create states
        self.states = []
        for state_name in self.state_names:
            if state_name in state_info:
                state = TimeLabel(state_name,
                                  description=state_info[state_name],
                                  group=self.group)
                setattr(self, state_name, state)
            else:
                state = TimeLabel(state_name, group=self.group)
                setattr(self, state_name, state)
            self.states.append(state)

    def __eq__(self, other):
        if isinstance(other, Labels):
            is_same = [
                self.group == other.group,
                self.group_info == other.group_info,
                set(self.state_names) == set(other.state_names),
                set(self.states) == set(other.states)
            ]

            return np.all(is_same)
        else:
            print(f"__eq__ is Not Implemented for {Labels} and {type(other)}")
            return NotImplemented

    def __str__(self):
        description = f"Label group : {self.group}\n"
        description = description + f"States:\n"
        for state_name in self.state_names:
            description = description + f"{getattr(self, state_name)}\n"
        return description

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_dict(cls, d: dict):
        """
        Create a Labels object from a dictionary.

        Returns:
            (Labels): a Labels object with attributes 'group', 'group_info', 'state_names', 'states'
                    filled from the dictionary fields.
        """
        group_info = None
        state_info = None
        if 'group_info' in d:
            group_info = d['group_info']
        if 'state_info' in d:
            state_info = d['state_info']
        return cls(d['group'], d['state_names'], group_info=group_info, state_info=state_info)

    @classmethod
    def from_df(cls, df: pd.DataFrame, group: str = None):
        """
        Create a Labels object from a dataframe.
                The dataframe must have columns 'group', 'name', optional column 'description'.
                'group' column must be the same for all rows. 'name' columns contains the state names,
                state names can repeat and only the unique state names will be used. The descriptions are optional,
                if provided then the descriptions of the same state name must be the same or left empty.

        Arg:
            df: the dataframe
            group: if not None, keep only the rows with this group name.
                Must be provided if there are multiple groups in the dataframe.
        Returns:
            (Labels): a Labels object with attributes 'group', 'group_info', 'state_names', 'states'
                    inferred and filled from the dataframe.
        """
        # if group is not none, keep only the group rows
        if group is not None:
            if group not in df['group'].unique():
                raise ValueError(f"Group {group} not found in the dataframe.")
            df = df[df['group'] == group]

        # check that group is the same for all rows
        group = df['group'].unique()
        if len(group) > 1:
            raise ValueError(
                f"More than one group found in the dataframe: {group}")
        group = group[0]

        state_names = df['name'].unique()
        state_info = {}
        for state_name in state_names:
            if 'description' in df.columns:
                # check that all descriptions are either the same or empty
                description = df[df['name'] == state_name]['description'].unique()
                if len(description) > 2:
                    raise ValueError(
                        f"More than one description found for state {state_name}: {description}")
                elif len(description) == 2:
                    # one must be an empty string or None
                    if not ('' in description or None in description):
                        raise ValueError(
                            f"More than one description found for state {state_name}: {description}")
                else:
                    state_info[state_name] = description[0]
            else:
                state_info[state_name] = None

        return cls(group, state_names.tolist(), group_info=None, state_info=state_info)


class Cycle:
    """
    Stores and preprocesses information about the repeated cycle of labels.
    Use it to create annotation when you have some periodic conditions.
    For example: light on , light off, light on, light off... will be made of list of labels [light_on, light_off]
    that repeat to cover the whole tie of the experiment. All labels must be from the same label group.
    Create multiple cycles to describe different label groups.

    Args:
        label_order: a list of labels in the right order in which they follow
        duration: duration of the corresponding labels, in frames (based on your imaging).
            Note that these are frames, not volumes !

    Attributes:
        name: the name of the cycle, the same as the name of the grou p of the labels.
        label_order: the order in which the labels follow in a cycle.
        duration: the duration of each label from the label_order ( in frames )
        cycle_length: the length of the cycle ( in frames )
        per_frame_list: mapping of frames to corresponding frames for one full cycle only.
    """

    def __init__(self, label_order: List[TimeLabel],
                 duration: Union[npt.NDArray, List[int]]):
        # check that all labels are from the same group
        label_group = label_order[0].group
        for label in label_order:
            assert label.group == label_group, \
                f"All labels should be from the same group, but got {label.group} and {label_group}"
        assert label_group is not None, \
            f"All labels should be from the same group, label group can not be None"

        # check that timing is int
        assert all(isinstance(n, (int, np.integer)) for n in
                   duration), "timing should be a list of int"

        self.name: str = label_group
        self.label_order: List[TimeLabel] = label_order
        self.duration: List[int] = list_of_int(duration)
        self.cycle_length: int = sum(self.duration)
        # list the length of the cycle, each element is the TimeLabel
        # TODO : turn it into an index ?
        self.per_frame_list: List[TimeLabel] = self._get_label_per_frame()

    def __eq__(self, other):
        if isinstance(other, Cycle):
            is_same = [
                self.name == other.name,
                self.label_order == other.label_order,
                self.duration == other.duration,
                self.cycle_length == other.cycle_length,
                self.per_frame_list == other.per_frame_list
            ]

            return np.all(is_same)
        else:
            print(f"__eq__ is Not Implemented for {Cycle} and {type(other)}")
            return NotImplemented

    def _get_label_per_frame(self) -> List[TimeLabel]:
        """
        Creates a list of labels per frame for one cycle only.

        Returns:
            labels per frame for one full cycle
        """
        per_frame_label_list = []
        for (label_time, label) in zip(self.duration, self.label_order):
            per_frame_label_list.extend(label_time * [label])
        return per_frame_label_list

    def __str__(self):
        description = f"Cycle : {self.name}\n"
        description = description + f"Length: {self.cycle_length}\n"
        for (label_time, label) in zip(self.duration, self.label_order):
            description = description + f"Label {label.name}: for {label_time} frames\n"
        return description

    def __repr__(self):
        return self.__str__()

    def fit_frames(self, n_frames: int) -> int:
        """
        Calculates how many cycles you need to fully cover n_frames.
        Assumes the cycle starts at the beginning of the recording.

        Args:
            n_frames: number of frames to cover, must be >= 0.

        Returns:
            number of cycles (n_cycles) necessary to cover n_frames:
            n_cycles*self.cycle_length >= n_frames
        """
        assert n_frames >= 0, "n_frames must be positive"
        n_cycles = int(np.ceil(n_frames / self.cycle_length))
        return n_cycles

    def fit_labels_to_frames(self, n_frames: int) -> List[TimeLabel]:
        """
        Create a list of labels corresponding to each frame in the range of n_frames

        Args:
            n_frames: number of frames to fit labels to, must be >= 0.

        Returns:
            labels per frame for each frame in range of n_frames
        """
        n_cycles = self.fit_frames(n_frames)
        label_per_frame_list = np.tile(self.per_frame_list, n_cycles)
        # crop the tail
        return list(label_per_frame_list[0:n_frames])

    def fit_cycles_to_frames(self, n_frames: int) -> List[int]:
        """
        Create a list of cycle ids (what cycle iteration it is) corresponding to each frame in the range of n_frames

        Args:
            n_frames: number of frames to fit cycle iterations to, must be >= 0.
        Returns:
            cycle id per frame for each frame in range of n_frames
        """
        n_cycles = self.fit_frames(n_frames)
        cycle_per_frame_list = []
        for n in np.arange(n_cycles):
            cycle_per_frame_list.extend([int(n)] * self.cycle_length)
        # crop the tail
        return cycle_per_frame_list[0:n_frames]

    def to_dict(self) -> dict:
        """
        Put all the information about a Cycle object into a dictionary.

        Returns:
            a dictionary with fields 'timing' and 'label_order' which store self.duration and self.label order.
        """
        label_order = [label.to_dict() for label in self.label_order]
        d = {'timing': self.duration, 'label_order': label_order}
        return d

    def to_json(self) -> str:
        """
        Put all the information about a Cycle object into a json file.

        Returns:
            a json with fields 'timing' and 'label_order' which store self.duration and self.label order.
        """
        return json.dumps(self.to_dict())

    def to_df(self, timing_conversion: Optional[dict] = None) -> pd.DataFrame:
        """
        Put all the information about a Cycle object into a dataframe.

        Args:
            timing_conversion: a dictionary to convert the timing into a different unit.
                For example, if you want to convert the timing from frames to seconds,
                and you were recording at 30 frames per second, you can use
                timing_conversion = {'frames': 1, 'seconds': 1/30}
                You can list multiple units in the dictionary, and the timing will be converted to all of them,
                for example if there are also 10 frames per volume, you can use:
                timing_conversion = {'frames': 1, 'seconds': 1/30, 'volumes': 1/10}
                You must include 'frames' in the dictionary! The value of frames does not have to be 1,
                but it must be consistent with the other units. the rest of the values.
                for example this is valid for the example above:
                timing_conversion = {'frames': 10, 'seconds': 1/3, 'volumes': 1}.
                if timing_conversion is None, then the timing is not converted.
                if timing_conversion is not None, then the timing is converted and both the original and converted
                timing are added to the dataframe.

        Returns:
            a dataframe with columns 'timing', 'group', 'name' and 'description'.
            'timing' will be written in all the units in the timing_conversion dictionary,
            or just in frames, if timing_conversion is None.
        """
        # TODO: move to/from methids to a separte class and inherit both Cycle and Timeline from it
        # prepare timing columns
        if timing_conversion is None:
            timing_conversion = {'frames': 1}
        assert 'frames' in timing_conversion.keys(), "frames must be in the timing_conversion dictionary"

        timing_columns = ['duration_' + unit for unit in timing_conversion.keys()]
        df = pd.DataFrame(columns=timing_columns + ['name', 'group', 'description'])

        # write timing columns
        for unit in timing_conversion.keys():
            frames_per_unit = timing_conversion['frames'] / timing_conversion[unit]
            duration = np.array(self.duration) / frames_per_unit
            # if all are integers, turn to integer
            if all(d.is_integer() for d in duration):
                duration = duration.astype(int)
            df['duration_' + unit] = duration

        # write labels
        df['name'] = [label.name for label in self.label_order]
        df['group'] = [label.group for label in self.label_order]
        df['description'] = [label.description for label in self.label_order]
        return df

    @classmethod
    def from_dict(cls, d: dict):
        """
        Create a Cycle object from a dictionary.

        Args:
            d: dictionary to initialize the cycle.

        Returns:
            (Cycle): a Cycle object with label_order and duration initialized from 'label_order' and
                    'timing' fields of the dictionary.
        """
        label_order = [TimeLabel.from_dict(ld) for ld in d['label_order']]
        return cls(label_order, d['timing'])

    @classmethod
    def from_json(cls, j: str):
        """
        Create a Cycle object from a json string.

        Args:
            j : json string to initialise the cycle

        Returns:
            (Cycle): a Cycle object with label_order and duration initialised from 'label_order' and
                    'timing' fields of the json srting.
        """
        d = json.loads(j)
        return cls.from_dict(d)

    @classmethod
    def from_df(cls, df: pd.DataFrame, timing_conversion: Optional[dict] = None):
        """
        Create a Cycle object from a dataframe.

        Args:
            df : dataframe to initialise the cycle.
                Must have columns 'group', 'name', optional column 'description'.
                Either column 'duration_frames' or duration column in any other unit
                and a timing_conversion dictionary to transform it to frames.
                For example if column 'duration_seconds' is present,
                the timing_conversion dictionary should be
                {'frames': 30, 'seconds': 1} if the recording was at 30 frames per second.
            timing_conversion: a dictionary to convert the timing into a different unit.

        Returns:
            (Cycle): a Cycle object with labels and duration initialised from 'group', 'name', 'description' and
                    duration fields of the dataframe. In the order in which they appear in the dataframe.
        """
        # TODO: add a check that the dataframe is valid
        # TODO: move to/from methids to a separte class and inherit both Cycle and Timeline from it
        label_order = []
        for _, row in df.iterrows():
            label_order.append(TimeLabel(row['name'],
                                         group=row['group'],
                                         description=row.get('description')
                                         )
                               )
        if 'duration_frames' in df.columns:
            duration_frames = df['duration_frames'].values
        else:
            assert timing_conversion is not None, "if duration_frames is not in the dataframe, " \
                                                  "timing_conversion must be provided"
            # check that timing_conversion is a dictionary
            assert isinstance(timing_conversion, dict), "timing_conversion must be a dictionary"
            # check that frames are present in the timing_conversion dictionary
            assert 'frames' in timing_conversion.keys(), "frames must be in the timing_conversion dictionary"
            # list units that are present in both dataframe and timing_conversion dictionary
            unit_list = [unit for unit in timing_conversion.keys() if 'duration_' + unit in df.columns]
            # check that at least one unit from the timing_conversion dictionary is present in the dataframe
            assert len(unit_list) > 0, "timing_conversion dictionary must have at least one " \
                                       "of the following keys in addition to 'frames': " + \
                                       ", ".join([col_name.split('_')[1] for col_name in
                                                  set(df.columns) - {'name', 'group', 'description'}])
            unit = unit_list[0]
            duration_frames = df['duration_' + unit].values * timing_conversion['frames'] / timing_conversion[unit]
            # if all values in duration are integer, convert to int, else raise an assertion
            if all(d.is_integer() for d in duration_frames):
                duration_frames = duration_frames.astype(int)
            else:
                assert False, "duration in frames must be integer after conversion from " + unit

        return cls(label_order, duration_frames)


class Timeline:
    """
    Information about the sequence of labels. Use it when you have non-periodic conditions.

    Args:
        label_order: a list of labels in the right order in which they follow
        duration: duration of the corresponding labels, in frames (based on your imaging). Note that these are
            frames, not volumes !
    """

    def __init__(self, label_order: List[TimeLabel],
                 duration: Union[npt.NDArray, List[int]]):

        # check that all labels are from the same group
        label_group = label_order[0].group
        for label in label_order:
            assert label.group == label_group, \
                f"All labels should be from the same group, but got {label.group} and {label_group}"
        assert label_group is not None, \
            f"All labels should be from the same group, label group can not be None"

        # check that timing is int
        assert all(isinstance(n, (int, np.integer)) for n in
                   duration), "timing should be a list of int"

        self.name = label_group
        self.label_order = label_order
        self.duration = list_of_int(duration)
        self.full_length = sum(self.duration)
        # list the length of the cycle, each element is the TimeLabel
        # TODO : turn it into an index ?
        self.per_frame_list = self.get_label_per_frame()

    def __eq__(self, other):
        if isinstance(other, Timeline):
            is_same = [
                self.name == other.name,
                self.label_order == other.label_order,
                self.duration == other.duration,
                self.full_length == other.full_length,
                self.per_frame_list == other.per_frame_list
            ]

            return np.all(is_same)
        else:
            print(
                f"__eq__ is Not Implemented for {Timeline} and {type(other)}")
            return NotImplemented

    def get_label_per_frame(self) -> List[TimeLabel]:
        """
        A list of labels per frame for the duration of the experiment.

        Returns:
            labels per frame for the experiment.
        """
        per_frame_label_list = []
        for (label_time, label) in zip(self.duration, self.label_order):
            per_frame_label_list.extend(label_time * [label])
        return per_frame_label_list

    def __str__(self):
        description = f"Timeline : {self.name}\n"
        description = description + f"Length: {self.full_length}\n"
        for (label_time, label) in zip(self.duration, self.label_order):
            description = description + f"Label {label.name}: for {label_time} frames\n"
        return description

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        """
        Put all the information about a Timeline object into a dictionary.

        Returns:
            a dictionary with fields 'label_order' and 'timing' .
        """
        return {'timing': self.duration,
                'label_order': [label.to_dict() for label in self.label_order],
                }

    def to_json(self) -> str:
        """
        Put all the information about a Timeline object into a json string.

        Returns:
            a json string with fields 'label_order' and 'timing' .
        """
        return json.dumps(self.to_dict())

    def to_df(self, timing_conversion: Optional[dict] = None) -> pd.DataFrame:
        """
        Put all the information about a Timeline object into a dataframe.

        Args:
            timing_conversion: a dictionary to convert the timing into a different unit.
                For example, if you want to convert the timing from frames to seconds,
                and you were recording at 30 frames per second, you can use
                timing_conversion = {'frames': 1, 'seconds': 1/30}
                You can list multiple units in the dictionary, and the timing will be converted to all of them,
                for example if there are also 10 frames per volume, you can use:
                timing_conversion = {'frames': 1, 'seconds': 1/30, 'volumes': 1/10}
                You must include 'frames' in the dictionary! The value of frames does not have to be 1,
                but it must be consistent with the other units. the rest of the values.
                for example this is valid for the example above:
                timing_conversion = {'frames': 10, 'seconds': 1/3, 'volumes': 1}.
                if timing_conversion is None, then the timing is not converted.
                if timing_conversion is not None, then the timing is converted and both the original and converted
                timing are added to the dataframe.

        Returns:
            a dataframe with columns 'timing', 'group', 'name' and 'description'.
            'timing' will be written in all the units in the timing_conversion dictionary,
            or just in frames, if timing_conversion is None.
        """
        # prepare timing columns
        if timing_conversion is None:
            timing_conversion = {'frames': 1}
        assert 'frames' in timing_conversion.keys(), "frames must be in the timing_conversion dictionary"

        timing_columns = ['duration_' + unit for unit in timing_conversion.keys()]
        df = pd.DataFrame(columns=timing_columns + ['name', 'group', 'description'])

        # write timing columns
        for unit in timing_conversion.keys():
            frames_per_unit = timing_conversion['frames'] / timing_conversion[unit]
            duration = np.array(self.duration) / frames_per_unit
            # if all are integers, turn to integer
            if all(d.is_integer() for d in duration):
                duration = duration.astype(int)
            df['duration_' + unit] = duration

        # write labels
        df['name'] = [label.name for label in self.label_order]
        df['group'] = [label.group for label in self.label_order]
        df['description'] = [label.description for label in self.label_order]
        return df

    @classmethod
    def from_dict(cls, d: dict):
        """
        Create a Timeline object from a dictionary.

        Args:
            d: dictionary to initialize the timeline.

        Returns:
            (Timeline): a Timeline object with label_order and duration initialized from 'label_order' and
                    'timing' fields of the dictionary.
        """
        label_order = [TimeLabel.from_dict(ld) for ld in d['label_order']]
        return cls(label_order, d['timing'])

    @classmethod
    def from_json(cls, j: str):
        """
        Create a Timeline object from a json string.

        Args:
            j : json string to initialise the cycle

        Returns:
            (Timeline): a Timeline object with label_order and duration initialised from 'label_order' and
                    'timing' fields of the json srting.
        """
        d = json.loads(j)
        return cls.from_dict(d)

    @classmethod
    def from_df(cls, df: pd.DataFrame, timing_conversion: Optional[dict] = None):
        """
        Create a Timeline object from a dataframe.

        Args:
            df : dataframe to initialise the timeline.
                Must have columns 'group', 'name', optional column 'description'.
                Either column 'duration_frames' or duration column in any other unit
                and a timing_conversion dictionary to transform it to frames.
                For example if column 'duration_seconds' is present,
                the timing_conversion dictionary should be
                {'frames': 30, 'seconds': 1} if the recording was at 30 frames per second.
            timing_conversion: a dictionary to convert the timing into a different unit.

        Returns:
            (Timeline): a Timeline object with labels and duration initialised from 'group', 'name', 'description' and
                    duration fields of the dataframe. In the order in which they appear in the dataframe.
        """
        label_order = []
        for _, row in df.iterrows():
            label_order.append(TimeLabel(row['name'],
                                         group=row['group'],
                                         description=row.get('description')
                                         )
                               )
        if 'duration_frames' in df.columns:
            duration_frames = df['duration_frames'].values
        else:
            assert timing_conversion is not None, "if duration_frames is not in the dataframe, " \
                                                  "timing_conversion must be provided"
            # check that timing_conversion is a dictionary
            assert isinstance(timing_conversion, dict), "timing_conversion must be a dictionary"
            # check that frames are present in the timing_conversion dictionary
            assert 'frames' in timing_conversion.keys(), "frames must be in the timing_conversion dictionary"
            # list units that are present in both dataframe and timing_conversion dictionary
            unit_list = [unit for unit in timing_conversion.keys() if 'duration_' + unit in df.columns]
            # check that at least one unit from the timing_conversion dictionary is present in the dataframe
            assert len(unit_list) > 0, "timing_conversion dictionary must have at least one " \
                                       "of the following keys in addition to 'frames': " + \
                                       ", ".join([col_name.split('_')[1] for col_name in
                                                  set(df.columns) - {'name', 'group', 'description'}])
            unit = unit_list[0]
            duration_frames = df['duration_' + unit].values * timing_conversion['frames'] / timing_conversion[unit]
            # if all values in duration are integer, convert to int, else raise an assertion
            if all(d.is_integer() for d in duration_frames):
                duration_frames = duration_frames.astype(int)
            else:
                assert False, "duration in frames must be integer after conversion from " + unit

        return cls(label_order, duration_frames)


class Annotation:
    """
    Time annotation of the experiment.

    Either frame_to_label_dict or n_frames need to be provided to infer the number of frames.
    If both are provided , they need to agree.

    Args:
        labels: Labels used to build the annotation
        info: a short description of the annotation
        frame_to_label: what label it is for each frame
        frame_to_cycle: what cycle it is for each frame
        cycle: for annotation from cycles keeps the cycle
        n_frames: total number of frames, will be inferred from frame_to_label if not provided
    """

    def __init__(self, n_frames: int, labels: Labels,
                 frame_to_label: List[TimeLabel], info: str = None,
                 cycle: Cycle = None, frame_to_cycle: List[int] = None):

        # get total experiment length in frames, check that it is consistent
        if frame_to_label is not None:
            assert n_frames == len(
                frame_to_label), f"The number of frames in the frame_to_label," \
                                 f"{len(frame_to_label)}," \
                                 f"and the number of frames provided," \
                                 f"{n_frames}, do not match."
        self.n_frames = n_frames
        self.frame_to_label = frame_to_label
        self.labels = labels
        self.name = self.labels.group
        self.info = info
        self.cycle = None

        # None if the annotation is not from a cycle
        assert (frame_to_cycle is None) == (
                cycle is None), "Creating Annotation: " \
                                "You have to provide both cycle and frame_to_cycle."
        # if cycle is provided , check the input and add the info
        if cycle is not None and frame_to_cycle is not None:
            # check that frame_to_cycle is int
            assert all(
                isinstance(n, (int, np.integer)) for n in
                frame_to_cycle), "frame_to_cycle should be a list of int"
            assert n_frames == len(
                frame_to_cycle), f"The number of frames in the frame_to_cycle," \
                                 f"{len(frame_to_cycle)}," \
                                 f"and the number of frames provided," \
                                 f"{n_frames}, do not match."
            self.cycle = cycle
            self.frame_to_cycle = frame_to_cycle

    def __eq__(self, other):
        if isinstance(other, Annotation):
            is_same = [
                self.n_frames == other.n_frames,
                self.frame_to_label == other.frame_to_label,
                self.labels == other.labels,
                self.name == other.name,
                self.info == other.info
            ]
            # if one of the annotations has a cycle but the other doesn't
            if (self.cycle is None) != (other.cycle is None):
                return False
            # if both have a cycle, compare cycles as well
            elif self.cycle is not None:
                is_same.extend([self.cycle == other.cycle,
                                self.frame_to_cycle == other.frame_to_cycle])
            return np.all(is_same)
        else:
            print(
                f"__eq__ is Not Implemented for {Annotation} and {type(other)}")
            return NotImplemented

    @classmethod
    def from_cycle(cls, n_frames: int, labels: Labels, cycle: Cycle,
                   info: str = None):
        """
        Creates an Annotation object from Cycle.

        Args:
            n_frames: total number of frames, must be provided
            labels: Labels used to build the annotation
            cycle: the cycle to create annotation from
            info: a short description of the annotation
        Returns:
            (Annotation): Annotation object
        """
        frame_to_label = cycle.fit_labels_to_frames(n_frames)
        frame_to_cycle = cycle.fit_cycles_to_frames(n_frames)
        return cls(n_frames, labels, frame_to_label, info=info,
                   cycle=cycle, frame_to_cycle=frame_to_cycle)

    @classmethod
    def from_timeline(cls, n_frames: int, labels: Labels, timeline: Timeline,
                      info: str = None):
        """
        Creates an Annotation object from Timeline.

        Args:
            n_frames: total number of frames, must be provided
            labels: Labels used to build the annotation
            timeline: the timeline to create annotation from
            info: a short description of the annotation
        Returns:
            (Annotation): Annotation object
        """
        assert n_frames == timeline.full_length, "number of frames and total timing should be the same"
        # make a fake cycle the length of the whole recording
        frame_to_label = timeline.per_frame_list
        return cls(n_frames, labels, frame_to_label, info=info)

    @classmethod
    def from_df(cls, n_frames: int, df: pd.DataFrame, is_cycle: bool = False, info: Optional[str] = None):
        """
        Creates an Annotation object from a dataframe.

        Args:
            n_frames: total number of frames, must be provided
            df: dataframe with columns 'frame' and 'label'
            is_cycle: if True, the annotation is for a cycle
            info: a short description of the annotation
        Returns:
            (Annotation): Annotation object
        """
        if is_cycle:
            cycle = Cycle.from_df(df)
            labels = Labels.from_df(df)
            return cls.from_cycle(n_frames, labels, cycle, info=info)
        else:
            timeline = Timeline.from_df(df)
            labels = Labels.from_df(df)
            return cls.from_timeline(n_frames, labels, timeline, info=info)

    def get_timeline(self) -> Timeline:
        """
        Transforms frame_to_label to Timeline

        Returns:
            timeline of the resulting annotation
        """
        duration = []
        labels = []
        for label, group in groupby(self.frame_to_label):
            duration.append(sum(1 for _ in group))
            labels.append(label)
        return Timeline(labels, duration)

    def cycle_info(self) -> str:
        """
        Creates and returns a description of a cycle.

        Returns:
            human-readable information about the cycle.
        """
        if self.cycle is None:
            cycle_info = "Annotation doesn't have a cycle"
        else:
            cycle_info = f"{self.cycle.fit_frames(self.n_frames)} full cycles" \
                         f" [{self.n_frames / self.cycle.cycle_length} exactly]\n"
            cycle_info = cycle_info + self.cycle.__str__()
        return cycle_info

    def __str__(self):
        description = f"Annotation type: {self.name}\n"
        if self.info is not None:
            description = description + f"{self.info}\n"
        description = description + f"Total frames : {self.n_frames}\n"
        return description

    def __repr__(self):
        return self.__str__()

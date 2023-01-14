# Create a new Experiment and save to a Database

Below is a quick example how-to create and experiment. For a line-by-line explanation, see the rest of the section.
```{.py3 .in py  linenums="1"}
### Imports

import vodex as vx

### Adding information about the Image Data

# the folder with the 3 movies of the toy dataset
data_dir = "data/test/test_movie"
frames_per_volume = 10
volume_m = vx.VolumeManager.from_dir(data_dir, frames_per_volume)

### Adding information about the Time Annotation

#### Create a Cycle

c_label = vx.Labels(# label type (group name), string
                    "c label",
                    # label names, keep this brief
                    # same rules as for variable names:
                    # no leading numbers, no spaces, no special characters
                    ["c1", "c2", "c3"],
                    # optional: info about the labels
                    # {label_name : label information}
                    state_info={"c1": "written c1", "c2": "written c1"})
c_label_cyc = vx.Cycle(# order of label presentation
                       # how they go in cycle
                       [c_label.c1, c_label.c2, c_label.c3],
                       # duration of each presentation  
                       [10,10,10])                                         

shape = vx.Labels("shape",
                   ["c", "s"],
                   state_info={"c": "circle on the screen", "s": "square on the screen"})
shape_cyc = vx.Cycle([shape.c, shape.s, shape.c],
                    [5,10,5])

#### Create a Timeline

light = vx.Labels("light",
                  ["on", "off"],
                  state_info={"on": "the intensity of the background is high",
                           "off": "the intensity of the background is low"},
                 # optional: information about the group, string
                  group_info="Light on inverts the colors")

light_tml = vx.Timeline(# order of label presentation
                        # how they go in WHOLE RECORDING
                        [light.off, light.on, light.off],
                        # duration of each presentation
                        # for timeline:
                        # the total duration MUST sum up
                        # to the total number of frames in the recording
                        [10,20,12])

#### Creating the annotations

n_frames = volume_m.n_frames # 42
c_label_an = vx.Annotation.from_cycle(  # the total number of frames in the recording
                                        n_frames,
                                        # the labels for the corresponding annotation
                                        c_label,
                                        # the annotation itself: a cycle, a timeline or a file
                                        c_label_cyc)
shape_an = vx.Annotation.from_cycle(n_frames, shape, shape_cyc)
light_an = vx.Annotation.from_timeline(n_frames, light, light_tml)

### Creating and Saving the Experiment

experiment = vx.Experiment.create(volume_m, [shape_an, c_label_an, light_an])
experiment.save("test.db")
```

## Imports
Import vodex:

```{.py3 .in py  linenums="3"}
import vodex as vx
```

## Adding information about the Image Data

You need to create a [VolumeManager](https://lemonjust.github.io/vodex/api/core/#src.vodex.core.VolumeManager) object, that will summarize and preprocess the information about the image data. To create one, you need to provide the following:

- the folder with the image data that you will work with,
- the type of the image files
- frames per volume ( if you work with a 2D data but still want to use vodex, set frames per volume to 1)
- first good frame ( OPTIONAL: if your recording was not synchronized with the volumes, you can specify the first frame in the recording that correspond to the beginning of a full volume )

Vodex will look into the folder and find all the files of the specified type. Vodex assumes that all the files are a recording of one continuous movie, so it is a good practice to store each imaging session in a separate folder. If you have to modify the order of the files or exclude some files from the movie, you can do so using [FileManager.change_files_order](https://lemonjust.github.io/vodex/api/core/#src.vodex.core.FileManager.change_files_order) (not shown in this example) Finally, vodex will use the frames per volume and first good frame to build a mapping of what frames correspond to which volume in the recording.

Create a VolumeManager object:
```{.py3 .in linenums="7"}
# the folder with the 3 movies of the toy dataset
data_dir = "data/test/test_movie"
frames_per_volume = 10
volume_m = vx.VolumeManager.from_dir(data_dir, frames_per_volume)
```
Inspect the located image files by looking at the FileManager:
``` {.py3 .in }
print(volume_m.file_manager)
```
``` {.text .yaml .no-copy }
files directory: data/test/test_movie
files [number of frames]:
0) mov0.tif [7]
1) mov1.tif [18]
2) mov2.tif [17]
```
Inspect the volume information:
```{.py3 .in }
print(volume_m.file_manager)
```
``` {.text .yaml .no-copy }
Total frames : 42
Volumes start on frame : 0
Total good volumes : 4
Frames per volume : 10
Tailing frames (not a full volume , at the end) : 2
```
Everything looks correct! We can move on to creating the experiment!

## Adding information about the Time Annotation

You need to create aa [Annotation](https://lemonjust.github.io/vodex/api/core/#src.vodex.core.Annotation) object, that will summarize and preprocess the information about the time annotation. You can have many annotations for the same data. The easiest way to create one, is by first creating a [Cycle](https://lemonjust.github.io/vodex/user_guide/core/#src.vodex.core.Cycle) , if the experimental conditions repeat through the experiment, or a [Timeline](https://lemonjust.github.io/vodex/user_guide/core/#src.vodex.core.Timeline), if they do not repeat. Cycles can describe a short period of time and will be repeated to cover the duration of the whole recording. Timelines must describe the whole recording.

### Create a Cycle
To create a Cycle, you need to provide:

- labels used to build the annotation (what type of conditions does this annotation describe),
- the order in which the labels are following in a cycle
- their duration of the conditions in the cycle ( in frames)

Create a Cycles to describe the  `c label` and the `shape` changes in the Toy Dataset.
First create the Labels:
```{.py3 .in py  linenums="16"}
c_label = vx.Labels(# label type (group name), string
                    "c label",
                    # label names, keep this brief
                    # same rules as for variable names:
                    # no leading numbers, no spaces, no special characters
                    ["c1", "c2", "c3"],
                    # optional: info about the labels
                    # {label_name : label information}
                    state_info={"c1": "written c1", "c2": "written c1"})
```
Now construct the Cycle. Note how you can use the names of the labels as the `c_label` attribute:
```{.py3 .in py  linenums="25"}
c_label_cyc = vx.Cycle(# order of label presentation
                       # how they go in cycle
                       [c_label.c1, c_label.c2, c_label.c3],
                       # duration of each presentation  
                       [10,10,10])
```
Inspect the information about the cycles:
```{.py3 .in}
print(c_label_cyc)
```
``` {.text .yaml .no-copy }
Cycle : c label
Length: 30
Label c1: for 10 frames
Label c2: for 10 frames
Label c3: for 10 frames
```
The above information is correct. Now, create the shape Cycle:
```{.py3 .in py  linenums="31"}
shape = vx.Labels("shape",
                   ["c", "s"],
                   state_info={"c": "circle on the screen", "s": "square on the screen"})
shape_cyc = vx.Cycle([shape.c, shape.s, shape.c],
                    [5,10,5])

```
```{.py3 .in }
print(c_label_cyc)
```
``` {.text .yaml .no-copy }
Cycle : shape
Length: 20
Label c: for 5 frames
Label s: for 10 frames
Label c: for 5 frames
```

### Create a Timeline
To create a Timeline, you need to provide:

- labels used to build the annotation (what type of conditions does this annotation describe),
- the order in which the labels are following in the whole recording
- their duration of the conditions in the cycle ( in frames). The total number of frames in the Timeline **must** be equal to the total number of frames in the recording.

Create a Timeline to describe the  `light` in the Toy Dataset:
First create the Labels:
```py  linenums="46"
light = vx.Labels("light",
                  ["on", "off"],
                  state_info={"on": "the intensity of the background is high",
                           "off": "the intensity of the background is low"},
                 # optional: information about the group, string
                  group_info="Light on inverts the colors")

light_tml = vx.Timeline(# order of label presentation
                        # how they go in WHOLE RECORDING
                        [light.off, light.on, light.off],
                        # duration of each presentation
                        # for timeline:
                        # the total duration MUST sum up
                        # to the total number of frames in the recording
                        [10,20,12])
```
Inspect the information about the timeline:
```{.py3 .in }
print(light_tml)
```
``` {.text .yaml .no-copy }
Timeline : light
Length: 42
Label off: for 10 frames
Label on: for 20 frames
Label off: for 12 frames
```

### Create a annotations
Use the cycles and the timeline that you created to create three time annotations. You will need to provide the total number of frames in the recording ( you can get it from the `volume_m`)

```py  linenums="57"
n_frames = volume_m.n_frames # 42
c_label_an = vx.Annotation.from_cycle(  # the total number of frames in the recording
                                        n_frames,
                                        # the labels for the corresponding annotation
                                        c_label,
                                        # the cycle
                                        c_label_cyc)
shape_an = vx.Annotation.from_cycle(n_frames, shape, shape_cyc)
light_an = vx.Annotation.from_timeline(n_frames, light, light_tml)
```
With this, you can create the Experiment!

## Creating and Saving the Experiment

The [Experiment](https://lemonjust.github.io/vodex/api/experiment/) will summarize the information about the Image Data and Annotations. It will create the Database that we can save and use it next time to initialize the Experiment with the exact same parameters. Experiment allows to search the frames based on volumes/ annotations and loads the image data using the appropriate [ImageLoader](https://lemonjust.github.io/vodex/api/core/#src.vodex.core.ImageLoader).

Create and save the experiment for the Toy Dataset:
```{.py3 .in py  linenums="69"}
experiment = vx.Experiment.create(volume_m, [shape_an, c_label_an, light_an])
experiment.save("test.db")
```
Alternatively, you can first create the experiment without any Time Annotation, and add it some later:
```{.py3 .in py  linenums="31"}
experiment = vx.Experiment.create(volume_m, [])
...
experiment.add_annotations([shape_an, c_label_an, light_an])
experiment.save("test.db")
```

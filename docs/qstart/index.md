# Quick start

This is a quick start example of how to use the `vodex` library.
If you need more information on volumetric functional imaging,
and library structure, refer to Guide.

The code is also available as a [jupyter notebook](https://github.com/LemonJust/vodex/blob/main/notebooks/01_create_experiment_and_load_volumes.ipynb).


## Create a new Experiment and save to DataBase

### Data
Get the test data from ???
TODO : make a function that grabs it!

### Imports

```{.py3 .in}
import vodex as vx

# to work with directories
from pathlib import Path

# for plotting :
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
```

### Adding information about the Volumes

You need to provide the folder with the data. Vodex will look into the folder, find all tiff files and assume it's one movie... so probably don't store any extra tiff files in that folder. It is generally a good idea to keed your raw data in a separate folder anyway. If you have to modify the order of the files or exclude some, yo ucan do too (not shown in this example, see FileManager class for more info on how to do it)
```python3
# data to create an experiment
TEST_DATA = "D:/Code/repos/vodex/data/test"
data_dir = Path(TEST_DATA, "test_movie")
```
Now give some information about the volumes. If your data is 2D - set frames_per_volume = 1 , it should work.
```{.py3 .in}
frames_per_volume = 10
volume_m = vx.VolumeManager.from_dir(data_dir, frames_per_volume)
volume_m
```
```{.text .out}
Total frames : 42
Volumes start on frame : 0
Total good volumes : 4
Frames per volume : 10
Tailing frames (not a full volume , at the end) : 2
```

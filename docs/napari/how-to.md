# How-To Guide

This folder contains instructions how to use napari-vodex.

# Launch napari-vodex

After following the installation instructions, you should see napari-vodex in your napari under `Plugis --> Vodex Data Loader (napari-vodex)`. When you start it, it creates two buttons - `Load Saved Experiment` and `Create New Experiment`

# Create New Experiment

Let's create the annotation for the toy dataset. Hit `Create New Experiment`. It will initialize three tabs:

The **Image Data** tab contains the information about the image files, their order and the information about the volumes : frames per volume and the first frame in the recording that is at the beginning of a volume.

The **Time Annotation** tab will contain the information about the time annotations ;)

The **Load/Save Data** tab allows to load individual volumes: based on their ID or based on the time annotation and also allows to save experiment (the information from the **Image Data** and **Time Annotation** tabs) to a database for future use with vodex or napari-vodex.

## 1. Image Data

Start with the Image Data tab.

1. Click `Browse` and choose the folder that contains your recording. The recording could be saved as multiple files.
2. Choose `file type`. Currently only TIFF files are supported, but if you added the support to your type of files, as described in [Contributions](https://lemonjust.github.io/vodex/contributions/), your file type should appear here.
3. Hit `Fetch files`. Vodex will search the directory for the files of the type that you choose. The files will appear in the window below.
4. **Inspect the files carefully!** The files appear in the order (top to bottom) in which they will be read by vodex. It is very important that they are in the correct order, or the rest of the annotations will be off. You can `drag the files to change the order` and delete the files that you don't want by clicking `Delete File` with a file selected.
Click `Save File Order` when you are done.
5. Enter `frames per volume`. If you work with a 2D data but still want to use vodex, leave frames per volume as 1.
6. If your recording was not synchronized with the volumes, you can specify the first frame in the recording that correspond to the beginning of a full volume (`first good frame`). It the beginning of the recording is at the beginning of a volume, leave it as 0.
7. Hit `Save Volume Info`.
8. **Read carefully** the short description of your data that: how many total frames, how many full volumes and how many frames that don't make a full volume at the beginning and at the end of the recording. If everything looks good, hit `Create Experiment`. This will allow you to load volumes using their ID numbers (For example volumes number 3, 8 and 30 to 40) and to add time annotations.

??? example "Toy Dataset: Creating experiment"

    The data used in these examples is a toy dataset, described in [Data](https://lemonjust.github.io/vodex/data/).
    Examples coming soon.

## 2. Time Annotation

Switch to Time Annotation tab and click `Add annotation`. You will be asked to create an annotation name: this is the name of the phenomena that you are describing. It could be some kind of stimuli/ drug/ behaviour, etc. After you name the annotation, the tab to enter the labels and the actual annotation of your recording.

1. Click `Add label` to create a label. A label is a particular condition: a certain state of the phenomena that you are describing. Add as many labels as you need. Double click on the description cell to add the description for each label (optional). You can delete a line by selecting it and clicking `Delete selected`.

2. Choose if the annotation is a `Cycle` or a `Timeline`. Cycles can describe a short period of time and will be repeated to cover the duration of the whole recording. Timelines must describe the whole recording.
3. Next you need to describe the order in which the conditions are following and their duration (in frames). Click `Add condition` to add a line to the annotation. You can delete a line by selecting it and clicking `Delete condition`.
4. After you are done, click `Add annotation to the experiment`. This will record the annotation and you can now use it at the `Load/Save Data` tab to choose the volumes that you want to load.

One experiment can have multiple annotations. Add as many annotations as you need.

??? example "Toy Dataset: Adding time annotation"

    Examples coming soon.

## 3. Save Experiment to a DataBase

Once you filled out the Image data information and added the annotations, you can save the experiment into a database.

Go to the `Load/Save Data` tab and click `Save` button at the bottom.
Choose the folder and give your database a name (*.db).
??? tip "How to see the DataBase content"

    We recommend using a DB Browser to open the content of the database file.


# Load Saved Experiment

To load the experiment, choose `Load Saved Experiment` after launching napari-vodex. Then click the `Load` button to search for the saved database file. Vodex will load the experiment and fill out all the `Image Data` and `Time Annotation` information, that you can see in the corresponding tabs. You can edit and delete time annotations and add more. These changes will not be saved to the original file, remember to save the file to keep the changes.

# Load Volumes into Napari Viewer

Finally, you can use napari-vodex to load volumes into napari. To do so you must have experiment created ( or loaded) and head to the `Load/Save Data` tab.

## Load based on volumes ID

Volume IDs are simply the numbers in which they follow from the beginning of the recording. The volumes numbering starts with 0: the first full volume's ID is 0, the second volume's ID is 1, the third's ID is 2 , etc. Enter the IDs of the volumes at the `Volumes:` edit line on top of the tab and press `Load`. Vodex will load the requested volumes and name them with the same text that you used to request them.

??? info "How to specify the volumes"

    You can specify the volumes as a line of integers separated by a comma (3, 4, 5, 6, 7) or request a slice by specifying the first and the last volume to load (3:7). Both examples will load the volumes with IDs 3, 4, 5, 6, and 7. You can mix the two methods. For example 2, 5:8, 3, 6:9 will load the volumes with IDs 2, 5, 6, 7, 8, 3, 6, 7, 8, 9 in this order. Note how the same volume can be loaded many times and the volume IDs do not have to be all ascending.

??? example "Toy Dataset: Loading volumes based on volume IDs"

    Examples coming soon.

## Load based on experimental conditions

To load volumes based on conditions, you must have at least one time annotation added to the experiment (make sure you pressed that `Add annotation to the experiment` button).

If you have added time annotations to the experiment, you will see the annotation's names and labels. `Click the checkboxes` by the labels for which you want to get the volumes and choose how to combine them with a logical `or` or a logical `and`. Then click `Find volumes` buttin to get a list of volume IDs that correspond to the chosen conditions, or  `Load` to load all such volumes into napari.

??? example "Toy Dataset: Loading volumes based on conditions"

    Examples coming soon.

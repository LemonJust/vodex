# Quick start

This is a quick start example of how to use the `vodex` library.
If you need more information on volumetric functional imaging,
and library structure, refer to Guide.
The data used in this tutorial is a [Toy Dataset](https://lemonjust.github.io/vodex/data/#toy-dataset).

Some code is also available as a [jupyter notebook](https://github.com/LemonJust/vodex/blob/main/notebooks/01_create_experiment_and_load_volumes.ipynb).


The core module is responsible for providing the core classes that vodex needs to manage the experimental data.
These classes are used to handle the various aspects of the data, such as the images themselves, the files they are stored in, and the annotations associated with them.
The Experiment class, uses these core classes to create and manage the database,
and provides methods for searching, loading, and saving the data.
The loaders module helps with reading and collecting information from specific file types, such as TIFF images.
The dbmethods module provides the functionality to interact with the database, such as writing and reading information.

Using the code allows for more flexibility, while the napari plugin provides a user-friendly interface for manual annotation and data organization.

Saving the data to a database allows to verify and share the information with others.
It also allows you to later load the information without having to re-enter it, making the process more efficient and less prone to errors.
The Experiment class has a save() method that can be used to save the information to a database file.
Once saved, the data can be loaded later using the load() method of the Experiment class, which initializes the Experiment object using the data from the database.

Once the data is saved to the database, you can use the Experiment.load() method to load it back and initialize an Experiment object,
then use the Experiment.choose_frames() method to select specific frames based on certain conditions,
and the Experiment.load_frames() method to load the image data for those frames.
You can also use the Experiment.add_annotations() method to add more annotations to the experiment later,
and the Experiment.save() method to save the updated experiment to the database again.

## Create Experiment
Create Experiment describes how to create a new experiment and save it to a database file for later use.

## Load Experiment
Load Experiment describes how to load a saved experiment from a database file.

## Load Volumes
Load Volumes describes how to load individual volumes with vodex.

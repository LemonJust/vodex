# Contributions

Contributions are very welcome. Tests can be run with tox, please ensure the coverage at least stays the same before you submit a pull request.

Don't know how to contribute? Here are some ides:

## Add support for new image types

Currently vodex only supports TIFF images, but you can add support for other image types.

I will try to create a more detailed description, but in a nutshell this can be done as follows:

To add support for a new image type, you will need to create a new loader class in [vodex.loaders](https://github.com/LemonJust/vodex/blob/main/src/vodex/loaders.py) module. Modify the [Loader](https://lemonjust.github.io/vodex/api/loaders/#src.vodex.loaders.Loader) class by filling out the implementation of `get_frame_dtype`, `get_frame_size`, `get_frames_in_file` and `load_frames` methods. It is important that these methods have the inputs and outputs as specified in Loader.

Once you have created the new class, you need to add your new image type, it's extensions and the new loader class that you implemented to the `VX_SUPPORTED_TYPES`, `VX_EXTENSION_TO_TYPE` and `VX_EXTENSION_TO_LOADER` dictionaries at the beginning of the [vodex.core](https://github.com/LemonJust/vodex/blob/main/src/vodex/core.py) module.

After this is done, vodex should be able to work with your image file!

## Add Experiment initialization from a .yaml config file

Currently a new experiment is initialized with a python script. This is sometimes not convenient, and it would be great to have an initialization from a .yaml config file as an option. You can modify [Experiment](https://github.com/LemonJust/vodex/blob/main/src/vodex/experiment.py) to add something like a `from_config` method (or whichever way you wish to name it) :)  

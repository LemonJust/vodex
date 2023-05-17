This part of the project documentation is a
reference for the technical implementation of the `VoDEx` project code.

VoDEx contains classes that assist in the creation, organization,
and storage of information related to image acquisition and
time annotation, allowing for the search and retrieval of image
data based on specific conditions. This functionality is split into
five modules: core, annotation, dbmethods, experiment, and
loaders.

- The core module provides the basic functionality for retrieving
image data information.

- The annotation module handles the construction, validation,
and storage of time annotation. For cyclic events, VoDEx
keeps track of cycle iterations, which is important in behavioral
experiments where the subject might become habituated to the
repeated stimulus or learn over the course of the experiment.

- The dbmethods module abstracts the SQL calls, providing an
easy-to-use interface to populate the SQLite database in which
VoDEx stores information, and to query the database.

- The loaders module contains classes designed to load image
data from specific file types, with current support for TIFF,
and allows for easy addition of support for other file formats (see [Contributions](https://lemonjust.github.io/vodex/contribute/) for more details).

- The experiment module contains the Experiment class,
connecting all the functionalities of the VoDEx package and
serving as the main point of entry for interacting with the
package.

# core
::: src.vodex.core
    options:
      members:
        - None
      show_root_heading: false
      show_source:

# experiment
::: src.vodex.experiment
    options:
      members:
        - None
      show_root_heading: false
      show_source: false

# annotation
::: src.vodex.annotation
    options:
      members:
        - None
      show_root_heading: false
      show_source:

# loaders
::: src.vodex.loaders
    options:
      members:
        - None
      show_root_heading: false
      show_source: false

# dbmethods
::: src.vodex.dbmethods
    options:
      members:
        - None
      show_root_heading: false
      show_source: false

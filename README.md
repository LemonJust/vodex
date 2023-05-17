# VoDEx: Volumetric Data and Experiment manager

[![License BSD-3](https://img.shields.io/pypi/l/vodex.svg?color=green)](https://github.com/LemonJust/vodex/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/vodex.svg?color=green)](https://pypi.org/project/vodex)
[![Python Version](https://img.shields.io/pypi/pyversions/vodex.svg?color=green)](https://python.org)
[![tests](https://github.com/LemonJust/vodex/workflows/tests/badge.svg)](https://github.com/LemonJust/vodex/actions)
[![codecov](https://codecov.io/gh/LemonJust/vodex/branch/main/graph/badge.svg)](https://codecov.io/gh/LemonJust/vodex)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-vodex)](https://napari-hub.org/plugins/napari-vodex)

VoDEx is an open-source Python library that streamlines the management and analysis of volumetric functional imaging data. It offers a suite of tools for creating, organizing, and storing information pertaining to image acquisition and time annotation. Additionally, it allows for the retrieval of imaging data based on specific experimental conditions, enabling researchers to access and analyze the data easily. VoDEx is available as both a standalone Python package and a napari plugin, providing a user-friendly solution for processing volumetric functional imaging data, even for researchers without extensive programming experience.

## Installation

You can install `vodex` via [pip](https://pypi.org/project/vodex):

    pip install vodex
    
or via conda:

    conda install vodex -c conda-forge

## Documentation

To get started with vodex, please refer to the [Documentation](https://lemonjust.github.io/vodex/). The documentation is continuously updated through the use of mkdocs and mkdocstrings packages, as well as GitHub actions, ensuring that any changes to the API are promptly reflected in the documentation.

## About

VoDEx is designed to address the challenges in functional imaging studies where accurate synchronization of the time course of experimental manipulations and stimulus presentations with resulting imaging data is crucial for analysis. It integrates the information about individual image frames, volumes, and experimental conditions and allows the retrieval of sub-portions of the 3D-time series datasets based on any of these identifiers. It logs all information related to the experiment into an SQLite database, enabling later data verification and sharing in accordance with the FAIR (Findable, Accessible, Interoperable, and Reusable) principles.

VoDEx is implemented as a [napari plugin](https://napari-hub.org/plugins/napari-vodex) for interactive use with a GUI and as an open-source [Python package](https://pypi.org/project/vodex), making it a useful tool for image analysis and allowing for integration into a wide range of analysis pipelines.

<p align="center">
  <img src="docs/assets/vodex_infographics_w_data_and_labels.PNG" alt="cover" width="1200"/>
</p>

## Use Cases
VoDEx has been successfully applied in the study of numerosity estimation in zebrafish larvae, where it played a key role in the processing of whole-brain functional imaging data acquired using light-sheet fluorescence microscopy. The implementation was carried out in Jupyter notebooks, as well as in a custom Python package specifically designed for this study, showcasing the versatility of integrating VoDEx into a comprehensive analysis pipeline. Full analysis pipelines for different numerosity stimuli combinations are available as sets of Jupyter notebooks at [github.com/LemonJust/numan](github.com/LemonJust/numan) under notebooks/individual datasets.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
`vodex` is free and open source software

## Citing VoDEx
If you use VoDEx in your research, please cite our paper:

Nadtochiy, A., Luu, P., Fraser, S. E., & Truong, T. V. (2023). VoDEx: a Python library for time annotation and management of volumetric functional imaging data. arXiv preprint arXiv:2305.07438.

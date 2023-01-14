# VoDEx: Volumetric Data and Experiment manager

[![License BSD-3](https://img.shields.io/pypi/l/vodex.svg?color=green)](https://github.com/LemonJust/vodex/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/vodex.svg?color=green)](https://pypi.org/project/vodex)
[![Python Version](https://img.shields.io/pypi/pyversions/vodex.svg?color=green)](https://python.org)
[![tests](https://github.com/LemonJust/vodex/workflows/tests/badge.svg)](https://github.com/LemonJust/vodex/actions)
[![codecov](https://codecov.io/gh/LemonJust/vodex/branch/main/graph/badge.svg)](https://codecov.io/gh/LemonJust/vodex)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-vodex)](https://napari-hub.org/plugins/napari-vodex)

A python library to load volumetric data based on experimental conditions.

## Installation

You can install `vodex` via [pip](https://pypi.org/project/vodex):

    pip install vodex

## Documentation

To get started with `vodex`, please refer to the [Documentation](https://lemonjust.github.io/vodex/).

## About

Recent advances in fluorescent microscopy and genetically-encoded calcium indicators made it possible to acquire large-scale 3D-time series datasets of brain activity. During these recordings, experimental conditions can vary: subjects can be presented with different stimuli and/or demonstrate different behavior. It is then required to annotate the data based on these experimental conditions and select the recording time of interest for the subsequent analysis. Data annotation is usually done manually or using a custom code deeply integrated with the analysis pipeline. Manual annotation is prone to error and is hard to document. Custom code often requires loading the whole dataset into the memory or depends on the exact file representation of data on a disc, which is not optimal for large datasets.

We introduce VoDEx, volumetric data and experiment manager, a data management tool that integrates the information about the individual image frames, volumes, volume slices, and experimental conditions and allows retrieval of sub-portions of the 3D-time series datasets based on any of these identifiers. It is implemented as a [napari plugin](https://napari-hub.org/plugins/napari-vodex) for interactive usage with a GUI and as an open-source [Python package](https://pypi.org/project/vodex) for easy inclusion into analysis pipelines.

<p align="center">
  <img src="img/cover.JPG" alt="cover" width="600"/>
</p>

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
`vodex` is free and open source software

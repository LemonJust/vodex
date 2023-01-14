# Examples

This section contains the links to the projects that used `vodex` in their analysis pipeline.

## NumAn: Numerosity analysis
NumAn is a growing collection of methods for neuron segmentation and calcium signal analysis. Existing tools for neural segmentation (Pachitariu et al., 2017, Giovannucci et al.,
2019) are optimized for 2D data and attempt to identify all active neurons in the recorded volume, which
is a hard computational task that requires data to have a high signal-to-noise ratio. The method that we
implemented in NumAn is based on the fMRI analysis: it generates a statistical parametric map in 3D to
highlight differences in the neural activity during each visual stimulus (Friston et al., 1994). One neuron
occupies multiple voxels in the image, all of which will show a similar response to stimuli. We cluster
the adjacent voxels with similar statistics to identify individual neurons and extract the signals, Figure 3.4.
Although this approach restricts the use of our tool to studies that focus on neural response to stimuli, it is more robust for data with a low signal-to-noise ratio, where other tools fail.

NumAn uses VoDEx to implement batch-processing of the large datasets and to manage complex time annotations.
Example of the NumAn pipeline can be found on [GitHub](https://github.com/LemonJust/numan).

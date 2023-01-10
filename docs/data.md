# Data

## Toy Dataset

![Toy Movie](assets/test_movie.gif){ align=left} To test vodex we are using a simple toy dataset, where each image frame is labeled with the **experimental conditions** as well as the **frame number** and volume **slice**.

You can download the toy dataset from [GitHub](https://github.com/LemonJust/vodex/tree/main/data/test/test_movie).
<br /> <br /><br /> <br /><br /> <br />

### Image Data structure

![Toy Movie](assets/data_frames.png){ align=left width=300} The toy dataset movie consists of **42 frames**.

**Each 10 frames make one volume**, so the whole recording consists of **4 full volumes** and **two extra frames** at the end of the recording.

The movie is split into **3 TIFF files**:

- 7 frames in the first file,
- 18 frames in the second file and
- 17 frames in the third one.

<br /> <br /><br />

### Time Annotation Structure

![Toy Movie](assets/data_annotation_a.png){ align=left width=300} We are tracking 3 types of toy experimental conditions in this dataset:

**1. c label**: a condition label that appears in the middle of the screen. It takes **c1, c2, c3** values. The movie starts with a c1 label for 10 frames, then c2 for 10 frames, followed by c3 for another 10 frames. This pattern repeats until the end of the recording.

| c label is a Cycle: |   |
| ----- | -------- |
| c1 | 10 frames |
| c2 | 10 frames |
| c3 | 10 frames |

**2. shape**: a circle or a square shapes on the screen. There is circle on the screen for the first 5 frames, then a square for 10 frames, then a circle again for 5 frames. This pattern repeats until the end of the recording.

| shape is a Cycle: |   |
| ----- | -------- |
| c | 5 frames |
| s | 10 frames |
| c | 5 frames |

![Toy Movie](assets/data_annotation_b.png){ align=left width=300}**3. light**: whether the background is bright or dark, with the light being **on** or **off** respectively. The light is off for the first 10 frames, then on for 20 frames and then off again for 12 frames.

| light is a Timeline: |   |
| ----- | -------- |
| off | 10 frames |
| on | 20 frames |
| off | 12 frames |

**<-- See a frame-by-frame labels in the image on the left [click on image to zoom in].**

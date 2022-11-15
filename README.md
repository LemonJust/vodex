# SfN2022 : please contact me on LemonJustGitHub@gmail.com with any questions :)

# vodex : Volumetric Data and Experiment manager
Vodex is a Python library for dealing with volumetric movies , written as a sequence of slices and split across multiple tif files.
It will keep track of full volumes / particular z-slices, making it easy to do requests like " give me all z slices # 23 in the whole movie" , or " give me volumes # 6, 85 and 54 ", regardless of whether these volumes are split between multiple tif files, or not. It can also link particular experimental conditions to certain volumes/slices, making it easy to request something like " give me all full volumes when I was shining the green light " or " give me all slices # 16 after we have administered the drug ".
<p align="center">
  <img src="img/cover.JPG" alt="cover" width="600"/>
</p>

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install vodex.

```bash
pip install vodex
```

## Usage

Please see notebooks/examples on [github](https://github.com/LemonJust/vodex).

## Contributing
Pull requests are welcome, but for now it's only me working on this project, so it might take me some time. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

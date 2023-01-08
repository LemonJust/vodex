# Installation

We recommend that you run these commands in a [virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## From PyPI

If you simply plan to use `vodex` as is, you can install the latest version via [pip](https://pypi.org/project/vodex/):
```bash
pip install vodex
```

## From source

If you wish to modify and contribute to `vodex` install it locally in editable mode.
This way, any changes you make to your copy of the package will reflect directly in your environment.

```bash
git clone https://github.com/LemonJust/vodex.git
cd vodex
pip install -e .
```

Note: Use [autoreload](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html) when working with packages in editable mode in IPython.

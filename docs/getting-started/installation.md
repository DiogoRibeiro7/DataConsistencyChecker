# Installation

DataConsistencyChecker supports **Python 3.10 to 3.14**. Its main dependencies are pandas, NumPy,
scikit-learn, SciPy, matplotlib, seaborn and IPython.

Both pandas 2 and pandas 3 are supported. CI runs the tests both with the locked dependency versions and
with the newest release of each dependency.

The package is not published on PyPI. Install it from GitHub, or from a clone.

## From GitHub

```bash
pip install "git+https://github.com/DiogoRibeiro7/DataConsistencyChecker.git"
```

This installs the `data_consistency_checker` package and the `data-consistency-checker` command.

## From a clone, with Poetry

The project uses [Poetry](https://python-poetry.org/) with a committed lock file, so every environment uses the
same dependency versions:

```bash
git clone https://github.com/DiogoRibeiro7/DataConsistencyChecker.git
cd DataConsistencyChecker
poetry install
```

Run code and commands in that environment with `poetry run`, for example
`poetry run data-consistency-checker --help`.

## Offline installation

Pre-download the wheels listed in `requirements.txt` on a machine with internet access, then install them
without an index:

```bash
pip install --no-index --find-links /path/to/wheels -r requirements.txt
```

## Checking the installation

```python
from data_consistency_checker import DataConsistencyChecker

print(len(DataConsistencyChecker(verbose=-1).get_test_list()), "checks available")
```

!!! note "Legacy import"
    `from check_data_consistency import DataConsistencyChecker` still works for existing notebooks, but new code
    should import from `data_consistency_checker`.

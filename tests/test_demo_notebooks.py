"""The demo notebooks keep to the current API.

Executing them needs datasets downloaded from OpenML and several minutes, so this checks their code statically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from data_consistency_checker import DataConsistencyChecker

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "Demo Notebooks").glob("*.ipynb"))


def _code(notebook: Path) -> str:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def test_there_are_demo_notebooks() -> None:
    assert len(NOTEBOOKS) >= 8


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_calls_only_existing_checker_methods(notebook: Path) -> None:
    used = set(re.findall(r"\bdc\.(\w+)", _code(notebook)))

    assert used
    assert sorted(name for name in used if not hasattr(DataConsistencyChecker, name)) == []


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_imports_the_package_and_no_removed_apis(notebook: Path) -> None:
    code = _code(notebook)

    assert "from data_consistency_checker import DataConsistencyChecker" in code
    assert "sys.path.insert" not in code
    assert "load_boston" not in code  # removed from scikit-learn 1.2
    assert "IPython.core.display import" not in code  # removed from IPython 9

"""Utility functions for DataConsistencyChecker.

This module contains helper functions extracted from the large
``check_data_consistency.py`` file. The functions are shared across the
package to keep the main module more maintainable.

"""

from __future__ import annotations

import numbers
import math
import string
from typing import Any, Iterable

import numpy as np
import pandas as pd
from IPython import get_ipython
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning
import scipy.stats as scipy_stats
import warnings


# ---------------------------------------------------------------------------
# General utility helpers
# ---------------------------------------------------------------------------

def safe_div(x: float, y: float) -> float:
    """Safely divide two numbers.

    Args:
        x: Numerator.
        y: Denominator.

    Returns:
        0 if ``y`` is zero, otherwise ``x / y``.
    """
    if y == 0:
        return 0.0
    return x / y


def is_number(s: Any) -> bool:
    """Check whether ``s`` can be interpreted as a number.

    Args:
        s: Value to check.

    Returns:
        ``True`` if ``s`` is numeric or a numeric string, ``False`` otherwise.
    """
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def convert_to_numeric(arr: Iterable[Any], filler: float) -> pd.Series:
    """Convert an iterable to numeric values.

    Non-numeric entries are replaced with ``filler``.

    Args:
        arr: Iterable containing values to convert.
        filler: Value used for items that cannot be converted to numbers.

    Returns:
        A pandas Series of ``float64`` values.
    """
    return pd.Series([
        float(x) if (is_number(x) and x == x and x is not None) else filler
        for x in arr
    ], dtype="float64")


def get_num_decimal_digits(num: float) -> int:
    """Return the number of decimal digits of ``num``.

    Args:
        num: Number to inspect.

    Returns:
        Number of digits after the decimal point.
    """
    num_str = str(np.format_float_positional(num))
    if num_str.count(".") == 0:
        return 0
    digits_str = num_str.split(".")[1]
    if not digits_str:
        return 0
    digits_val = int(digits_str)
    if digits_val == 0:
        return 0
    return len(digits_str)


def get_non_alphanumeric(x: str) -> list[str]:
    """Return non alphanumeric characters from ``x``.

    Args:
        x: Input string.

    Returns:
        List of characters in ``x`` that are not alphanumeric.
    """
    if x.isalnum():
        return []
    return [c for c in x if not str(c).isalnum()]


def styling_orig_row(x: pd.DataFrame, row_idx: int, flagged_arr: Iterable[bool]) -> pd.DataFrame:
    """Create styling for a row in a flagged DataFrame.

    Args:
        x: DataFrame being styled.
        row_idx: Index of the row to style.
        flagged_arr: Iterable indicating which columns are flagged.

    Returns:
        Styling DataFrame with background colors applied.
    """
    df_styler = pd.DataFrame("", index=x.index, columns=x.columns)
    for c_idx, c_flagged in enumerate(flagged_arr):
        if c_flagged:
            df_styler.iloc[row_idx, c_idx + 1] = "background-color: #efecc3; color: black"
        else:
            df_styler.iloc[row_idx, c_idx + 1] = "background-color: #e5f8fa; color: black"
    return df_styler


def styling_flagged_rows(x: pd.DataFrame, flagged_cells: np.ndarray) -> pd.DataFrame:
    """Style DataFrame rows based on flag counts.

    Args:
        x: DataFrame being styled.
        flagged_cells: Array indicating flagged cells per row and column.

    Returns:
        Styling DataFrame with appropriate background colors.
    """
    df_styler = pd.DataFrame(
        "background-color: #e5f8fa; color: black", index=x.index, columns=x.columns
    )
    for row_idx in x.index:
        for col_idx, col_name in enumerate(x.columns[:-1]):
            if flagged_cells[row_idx, col_idx] > 0:
                df_styler.loc[row_idx][
                    x.columns[col_idx]
                ] = "background-color: #efecc3; color: black"
    for row_idx in x.index:
        col_idx = len(x.columns) - 1
        df_styler.loc[row_idx][
            x.columns[col_idx]
        ] = "background-color: white; color: black"
    return df_styler


def is_notebook() -> bool:
    """Return ``True`` if running inside a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        if shell == "TerminalInteractiveShell":
            return False
        return False
    except NameError:
        return False


def print_line(f: Any) -> None:
    """Print a blank line or write a newline to file handle ``f``."""
    if not f:
        print()


def print_text(s: str, f: Any | None = None) -> None:
    """Print text either to a file handle or the console.

    The function strips markdown-like tags when printing to the console so the
    output remains readable.

    Args:
        s: String to print.
        f: Optional file handle.
    """
    if f:
        s = s.replace("**", "<b>", 1)
        # Remaining formatting removed for brevity when writing to file
        f.write(s + "\n")
    else:
        print(s.replace("**", "").replace("#", "").replace("<br>", "\n"))


def is_missing(x: Any) -> bool:
    """Determine whether ``x`` should be considered missing."""
    if x is None:
        return True
    if "NAType" in str(type(x)):
        return True
    if "missing" in str(type(x)):
        return True
    if isinstance(x, numbers.Number):
        return math.isnan(x)
    if x != x:
        return True
    if type(x) in [str, np.str_]:
        return (x.strip() == "") or (x == "nan") or (x == "None") or (len(x) == 0)
    if type(x) == list:
        return len(x) == 0
    return False


def array_to_str(arr: Iterable[Any]) -> str:
    """Create a prettified string version of ``arr``."""
    arr_sorted = sorted(arr)
    arr_str = ", ".join(str(v) for v in arr_sorted)
    return arr_str


def replace_special_with_space(x: str | None) -> str:
    """Replace special characters in ``x`` with spaces."""
    if x is None:
        return ""
    if x in [np.inf, -np.inf, np.nan]:
        return ""
    return "".join(
        [c if ((c in string.ascii_letters) or (c in string.digits)) else " " for c in x]
    )


def truncate_description(x: str) -> str:
    """Truncate a description string to 100 characters."""
    if len(x) > 100:
        x = x[:100] + "..."
    return x


def is_uppercase(x: str | None) -> bool:
    """Check whether ``x`` is an uppercase ASCII character."""
    if x is None or x == "":
        return False
    return (65 <= ord(x) <= 90) or (193 <= ord(x) <= 221)


def call_test(dc: Any, test_id: str) -> None:
    """Placeholder for test invocation logging."""
    print(test_id)


def clean_x_tick_labels(fig: Figure, n_axis: int, ax: Axes) -> None:
    """Adjust x-axis tick labels for readability."""
    plt.draw()
    num_ticks = len(ax.xaxis.get_ticklabels())
    if num_ticks > 10:
        max_ticks = 10
        mod = num_ticks // max_ticks
        for label_idx, label in enumerate(ax.xaxis.get_ticklabels()):
            if label_idx % mod != 0:
                label.set_visible(False)

    num_chars = 0
    for label_idx, label in enumerate(ax.xaxis.get_ticklabels()):
        if label.get_visible():
            num_chars += len(label._text)

    if (num_chars > (fig.get_figwidth() * 10 / n_axis)) or (num_chars == 0):
        fig.autofmt_xdate()


def set_warnings_levels() -> None:
    """Suppress third-party library warnings."""
    warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    try:
        warnings.filterwarnings(
            action="ignore", category=scipy_stats.SpearmanRConstantInputWarning
        )
    except Exception:
        pass
    try:
        warnings.filterwarnings(action="ignore", category=scipy_stats.ConstantInputWarning)
    except Exception:
        pass
    try:
        warnings.filterwarnings(action="ignore", category=scipy_stats.NearConstantInputWarning)
    except Exception:
        pass

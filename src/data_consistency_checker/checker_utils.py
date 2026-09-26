"""Utility functions for DataConsistencyChecker.

This module contains helper functions extracted from the large
``check_data_consistency.py`` file. The functions are shared across the
package to keep the main module more maintainable.

"""

from __future__ import annotations

import contextlib
import math
import numbers
import string
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
from IPython import get_ipython
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning

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
        for col_idx, _col_name in enumerate(x.columns[:-1]):
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
    if type(x) is list:
        return len(x) == 0
    return False


def array_to_str(arr: Iterable[Any]) -> str:
    """Create a prettified string version of ``arr``."""
    arr_sorted = sorted(arr)
    return ", ".join(str(v) for v in arr_sorted)


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
    for label in ax.xaxis.get_ticklabels():
        if label.get_visible():
            num_chars += len(label._text)

    if (num_chars > (fig.get_figwidth() * 10 / n_axis)) or (num_chars == 0):
        fig.autofmt_xdate()


def set_warnings_levels() -> None:
    """Suppress third-party library warnings."""
    warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    with contextlib.suppress(Exception):
        warnings.filterwarnings(
            action="ignore", category=scipy_stats.SpearmanRConstantInputWarning
        )
    with contextlib.suppress(Exception):
        warnings.filterwarnings(action="ignore", category=scipy_stats.ConstantInputWarning)
    with contextlib.suppress(Exception):
        warnings.filterwarnings(action="ignore", category=scipy_stats.NearConstantInputWarning)


# ---------------------------------------------------------------------------
# pandas version compatibility
# ---------------------------------------------------------------------------

def as_str(values: pd.Series) -> pd.Series:
    """Convert values to strings, including missing values.

    Before pandas 3, ``astype(str)`` also converted missing values to strings (``'nan'``, ``'None'``,
    ``'NaT'``), and the checks were written for that. pandas 3 keeps them missing, so this converts them
    as earlier versions did, giving the same strings whatever the pandas version.

    Args:
        values: The values to convert.

    Returns:
        An object Series of strings, with the index and name of ``values``.
    """
    strings = np.array(values.astype(str), dtype=object)
    missing = pd.isna(strings)
    strings[missing] = [str(v) for v in np.array(values, dtype=object)[missing]]
    return pd.Series(strings, index=values.index, name=values.name, dtype=object)


def map_elements(df: pd.DataFrame, func: Callable[[Any], Any]) -> pd.DataFrame:
    """Apply ``func`` to each element of ``df``.

    ``DataFrame.applymap()`` was renamed ``DataFrame.map()`` in pandas 2.1 and removed in pandas 3.

    Args:
        df: The data to map.
        func: The function applied to each element.

    Returns:
        A DataFrame of the results, with the same shape as ``df``.
    """
    return df.map(func) if hasattr(df, "map") else df.applymap(func)


def column_from_values(values: list[Any], index: pd.Index) -> pd.Series:
    """Build a column from Python values, inferring its dtype as pandas 2 did.

    pandas 3 infers its string dtype for text, which turns ``None`` into NaN. Text is kept in an object column
    instead, as before.

    Args:
        values: The column's values.
        index: The column's index.

    Returns:
        A Series of ``values`` with the inferred dtype, or object dtype for text.
    """
    column = pd.Series(values, index=index)
    return pd.Series(values, index=index, dtype=object) if isinstance(column.dtype, pd.StringDtype) else column


def normalise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the dtypes the checks were written for.

    pandas 3 stores text in a dedicated string dtype and creates datetimes at microsecond resolution. The checks
    expect text in object columns, with NaN for missing values, and datetimes in nanoseconds (the pandas 2
    defaults), so such columns are converted.

    Args:
        df: The data to convert.

    Returns:
        A copy of ``df`` with string columns as object and datetime columns in nanoseconds.
    """
    df = df.copy()
    for col_idx in range(df.shape[1]):
        col = df.iloc[:, col_idx]
        if isinstance(col.dtype, pd.StringDtype):
            values = col.to_numpy(dtype=object, na_value=np.nan)
            df.isetitem(col_idx, pd.Series(values, index=df.index, dtype=object))
        elif pd.api.types.is_datetime64_any_dtype(col.dtype) and col.dt.unit != "ns":
            df.isetitem(col_idx, col.dt.as_unit("ns"))
    return df

"""Command-line interface for DataConsistencyChecker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from .checker import DataConsistencyChecker


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Load a supported tabular input file."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(
        f"Unsupported input format {suffix!r}. Use CSV, TSV, JSON, or JSONL."
    )


def _write_report(path: Path, payload: dict[str, object]) -> None:
    """Write a structured report as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_test_ids(
    checker: DataConsistencyChecker,
    test_ids: Sequence[str] | None,
) -> None:
    """Validate explicit test IDs before executing the checker."""
    if not test_ids:
        return
    valid = set(checker.get_test_list())
    unknown = sorted(set(test_ids) - valid)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown or unsupported test ID(s): {joined}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data-consistency-checker",
        description="Run interpretable data-consistency and outlier checks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list-tests",
        help="List implemented consistency checks.",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the test list as JSON.",
    )

    check_parser = subparsers.add_parser(
        "check",
        help="Run checks on a CSV, TSV, JSON, or JSONL file.",
    )
    check_parser.add_argument("input", type=Path, help="Input dataset path.")
    check_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the structured report to this JSON file.",
    )
    filters = check_parser.add_mutually_exclusive_group()
    filters.add_argument(
        "--tests",
        nargs="+",
        metavar="TEST_ID",
        help="Run only the specified test IDs.",
    )
    filters.add_argument(
        "--exclude-tests",
        nargs="+",
        metavar="TEST_ID",
        help="Run all implemented tests except the specified IDs.",
    )
    check_parser.add_argument(
        "--date-column",
        action="append",
        default=[],
        help="Treat this column as a date column. Repeat for multiple columns.",
    )
    check_parser.add_argument(
        "--fast-only",
        action="store_true",
        help="Run only checks marked as fast.",
    )
    check_parser.add_argument(
        "--max-combinations",
        type=int,
        default=100_000,
        help="Maximum number of feature combinations to evaluate.",
    )
    check_parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Stop immediately if a consistency-test implementation fails.",
    )
    check_parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=(-1, 0, 1, 2),
        default=0,
        help="Checker verbosity level (-1 through 2).",
    )
    return parser


def _run_list_tests(*, as_json: bool) -> int:
    checker = DataConsistencyChecker(verbose=-1)
    test_ids = checker.get_test_list()
    if as_json:
        print(json.dumps(test_ids, indent=2))
    else:
        for test_id in test_ids:
            print(test_id)
    return 0


def _run_check(args: argparse.Namespace) -> int:
    checker = DataConsistencyChecker(
        verbose=args.verbose,
        max_combinations=args.max_combinations,
    )
    _validate_test_ids(checker, args.tests)
    _validate_test_ids(checker, args.exclude_tests)

    dataframe = _load_dataframe(args.input)
    checker.init_data(
        dataframe,
        known_date_cols=args.date_column or None,
    )
    checker.check_data_quality(
        execute_list=args.tests,
        exclude_list=args.exclude_tests,
        fast_only=args.fast_only,
        raise_on_error=args.raise_on_error,
    )

    report = checker.get_report()
    payload = report.to_dict()
    if args.output is not None:
        _write_report(args.output, payload)

    print(
        f"rows={report.n_rows} columns={report.n_columns} "
        f"tests={len(report.executed_tests)} patterns={len(report.patterns)} "
        f"exceptions={len(report.exceptions)} failures={len(report.execution_failures)}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DataConsistencyChecker command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-tests":
            return _run_list_tests(as_json=args.json)
        if args.command == "check":
            return _run_check(args)
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())

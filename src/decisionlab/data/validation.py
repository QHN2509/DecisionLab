"""Load, validate, and summarize the original choices13k data files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from decisionlab.data.fetch import (
    DEFAULT_DESTINATION,
    DEFAULT_MANIFEST,
    load_manifest,
    sha256_file,
)

CSV_COLUMNS = (
    "Problem",
    "Feedback",
    "n",
    "Block",
    "Ha",
    "pHa",
    "La",
    "Hb",
    "pHb",
    "Lb",
    "LotShapeB",
    "LotNumB",
    "Amb",
    "Corr",
    "bRate",
    "bRate_std",
)

FEATURE_TYPES = {
    "Problem": "integer",
    "Feedback": "boolean",
    "n": "integer",
    "Block": "integer",
    "Ha": "integer",
    "pHa": "float",
    "La": "integer",
    "Hb": "integer",
    "pHb": "float",
    "Lb": "integer",
    "LotShapeB": "integer/category",
    "LotNumB": "integer",
    "Amb": "boolean",
    "Corr": "integer/category",
    "bRate": "float/target",
    "bRate_std": "float/measurement",
}

STRUCTURAL_FIELDS = (
    "ha",
    "pha",
    "la",
    "hb",
    "phb",
    "lb",
    "lot_shape_b",
    "lot_num_b",
    "amb",
    "corr",
)


class DataValidationError(ValueError):
    """Raised when choices13k violates its declared data contract."""


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    """One validated condition row from c13k_selections.csv."""

    problem: int
    feedback: bool
    n: int
    block: int
    ha: int
    pha: float
    la: int
    hb: int
    phb: float
    lb: int
    lot_shape_b: int
    lot_num_b: int
    amb: bool
    corr: int
    brate: float
    brate_std: float


def _parse_bool(value: str, field: str, row_number: int) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise DataValidationError(
        f"row {row_number}: {field} must be exactly 'True' or 'False', found {value!r}"
    )


def _parse_row(row: dict[str, str], row_number: int) -> SelectionRecord:
    missing = [name for name in CSV_COLUMNS if row.get(name, "").strip() == ""]
    if missing:
        raise DataValidationError(f"row {row_number}: missing values in {missing}")
    try:
        record = SelectionRecord(
            problem=int(row["Problem"]),
            feedback=_parse_bool(row["Feedback"], "Feedback", row_number),
            n=int(row["n"]),
            block=int(row["Block"]),
            ha=int(row["Ha"]),
            pha=float(row["pHa"]),
            la=int(row["La"]),
            hb=int(row["Hb"]),
            phb=float(row["pHb"]),
            lb=int(row["Lb"]),
            lot_shape_b=int(row["LotShapeB"]),
            lot_num_b=int(row["LotNumB"]),
            amb=_parse_bool(row["Amb"], "Amb", row_number),
            corr=int(row["Corr"]),
            brate=float(row["bRate"]),
            brate_std=float(row["bRate_std"]),
        )
    except ValueError as error:
        raise DataValidationError(f"row {row_number}: invalid feature type: {error}") from error
    numeric_values = (
        record.pha,
        record.phb,
        record.brate,
        record.brate_std,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise DataValidationError(f"row {row_number}: non-finite numeric value")
    if record.problem <= 0 or record.n <= 0:
        raise DataValidationError(f"row {row_number}: Problem and n must be positive")
    if record.block not in {1, 2, 3, 4, 5}:
        raise DataValidationError(f"row {row_number}: Block must be in 1..5")
    if not 0.0 <= record.pha <= 1.0 or not 0.0 <= record.phb <= 1.0:
        raise DataValidationError(f"row {row_number}: probabilities must be in [0, 1]")
    if record.lot_shape_b not in {0, 1, 2, 3} or not 1 <= record.lot_num_b <= 8:
        raise DataValidationError(f"row {row_number}: invalid Gamble B lottery structure")
    if (record.lot_shape_b == 0) != (record.lot_num_b == 1):
        raise DataValidationError(
            f"row {row_number}: undefined shape must mean one lottery outcome"
        )
    if record.corr not in {-1, 0, 1}:
        raise DataValidationError(f"row {row_number}: Corr must be -1, 0, or 1")
    if not 0.0 <= record.brate <= 1.0:
        raise DataValidationError(f"row {row_number}: bRate must be in [0, 1]")
    if not 0.0 <= record.brate_std <= 1.0:
        raise DataValidationError(f"row {row_number}: bRate_std must be in [0, 1]")
    if (not record.feedback and record.block != 1) or (record.feedback and record.block == 1):
        raise DataValidationError(f"row {row_number}: Feedback is inconsistent with Block")
    return record


def _record_tuple(record: SelectionRecord) -> tuple[Any, ...]:
    return tuple(asdict(record).values())


def _structural_tuple(record: SelectionRecord) -> tuple[Any, ...]:
    return tuple(getattr(record, name) for name in STRUCTURAL_FIELDS)


def load_selections(path: Path) -> list[SelectionRecord]:
    """Load and validate the selections CSV without altering it."""
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise DataValidationError(
                f"unexpected CSV schema: expected {CSV_COLUMNS}, found {reader.fieldnames}"
            )
        records = [_parse_row(row, row_number) for row_number, row in enumerate(reader, start=2)]
    if not records:
        raise DataValidationError("selections CSV is empty")
    duplicate_count = len(records) - len({_record_tuple(record) for record in records})
    if duplicate_count:
        raise DataValidationError(f"found {duplicate_count} exact duplicate selection rows")

    grouped: dict[int, list[SelectionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.problem].append(record)
    for problem, problem_rows in grouped.items():
        if len(problem_rows) > 2:
            raise DataValidationError(f"Problem {problem} occurs more than twice")
        if len({_structural_tuple(record) for record in problem_rows}) != 1:
            raise DataValidationError(f"Problem {problem} changes structure across rows")
        if len({record.feedback for record in problem_rows}) != len(problem_rows):
            raise DataValidationError(f"Problem {problem} repeats a feedback condition")
    return records


def load_problems(path: Path, selections: list[SelectionRecord]) -> dict[str, Any]:
    """Load and validate row-indexed gamble descriptions against selections."""
    with path.open(encoding="utf-8") as source:
        problems = json.load(source)
    if not isinstance(problems, dict):
        raise DataValidationError("problems JSON must be an object")
    expected_keys = {str(index) for index in range(len(selections))}
    if set(problems) != expected_keys:
        raise DataValidationError(
            "problems JSON keys must be contiguous zero-based CSV row indexes"
        )

    descriptions_by_problem: dict[int, Any] = {}
    for index, selection in enumerate(selections):
        problem = problems[str(index)]
        if not isinstance(problem, dict) or set(problem) != {"A", "B"}:
            raise DataValidationError(f"JSON row {index}: expected exactly A and B")
        expected_evs = {
            "A": selection.pha * selection.ha + (1.0 - selection.pha) * selection.la,
            "B": selection.phb * selection.hb + (1.0 - selection.phb) * selection.lb,
        }
        for option in ("A", "B"):
            outcomes = problem[option]
            if not isinstance(outcomes, list) or not outcomes:
                raise DataValidationError(f"JSON row {index}: {option} must contain outcomes")
            probability_sum = 0.0
            expected_value = 0.0
            for outcome in outcomes:
                if not isinstance(outcome, list) or len(outcome) != 2:
                    raise DataValidationError(
                        f"JSON row {index}: {option} outcomes must be [probability, payoff]"
                    )
                probability, payoff = outcome
                if (
                    isinstance(probability, bool)
                    or isinstance(payoff, bool)
                    or not isinstance(probability, int | float)
                    or not isinstance(payoff, int | float)
                    or not math.isfinite(probability)
                    or not math.isfinite(payoff)
                    or not 0.0 <= probability <= 1.0
                ):
                    raise DataValidationError(f"JSON row {index}: invalid {option} outcome")
                probability_sum += probability
                expected_value += probability * payoff
            if not math.isclose(probability_sum, 1.0, abs_tol=1e-9):
                raise DataValidationError(
                    f"JSON row {index}: {option} probabilities do not sum to 1"
                )
            if not math.isclose(expected_value, expected_evs[option], abs_tol=1e-9):
                raise DataValidationError(
                    f"JSON row {index}: {option} expected value disagrees with CSV"
                )
        existing = descriptions_by_problem.setdefault(selection.problem, problem)
        if existing != problem:
            raise DataValidationError(
                f"JSON row {index}: Problem {selection.problem} changes across conditions"
            )
    return problems


def summarize_dataset(
    selections: list[SelectionRecord],
    problems: dict[str, Any],
    file_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a concise, JSON-serializable audit summary."""
    problem_counts = Counter(record.problem for record in selections)
    rows_by_problem: dict[int, list[SelectionRecord]] = defaultdict(list)
    structural_problem_ids: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for record in selections:
        rows_by_problem[record.problem].append(record)
        structural_problem_ids[_structural_tuple(record)].add(record.problem)
    paired_problem_ids = sum(len(rows) == 2 for rows in rows_by_problem.values())
    feedback_only_problem_ids = sum(
        len(rows) == 1 and rows[0].feedback for rows in rows_by_problem.values()
    )
    no_feedback_only_problem_ids = sum(
        len(rows) == 1 and not rows[0].feedback for rows in rows_by_problem.values()
    )
    feedback_counts = Counter(str(record.feedback) for record in selections)
    sample_counts = [record.n for record in selections]
    brates = [record.brate for record in selections]
    return {
        "selection_rows": len(selections),
        "json_rows": len(problems),
        "unique_problem_ids": len(problem_counts),
        "problem_id_range": [min(problem_counts), max(problem_counts)],
        "problem_row_multiplicity": {
            str(count): frequency
            for count, frequency in sorted(Counter(problem_counts.values()).items())
        },
        "problem_condition_coverage": {
            "paired_feedback_and_no_feedback": paired_problem_ids,
            "feedback_only": feedback_only_problem_ids,
            "no_feedback_only": no_feedback_only_problem_ids,
        },
        "structural_fingerprints": {
            "unique": len(structural_problem_ids),
            "shared_across_problem_ids": sum(
                len(problem_ids) > 1 for problem_ids in structural_problem_ids.values()
            ),
        },
        "missing_values": {column: 0 for column in CSV_COLUMNS},
        "exact_duplicate_rows": 0,
        "feature_types": FEATURE_TYPES,
        "bRate": {
            "min": min(brates),
            "max": max(brates),
            "zero_count": sum(value == 0.0 for value in brates),
            "one_count": sum(value == 1.0 for value in brates),
        },
        "sample_counts_n": {
            "min": min(sample_counts),
            "max": max(sample_counts),
            "mean": statistics.fmean(sample_counts),
            "median": statistics.median(sample_counts),
            "sum_across_rows": sum(sample_counts),
        },
        "feedback_rows": dict(sorted(feedback_counts.items())),
        "sha256": file_hashes or {},
    }


def load_and_validate(
    raw_dir: Path = DEFAULT_DESTINATION,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Verify source hashes, validate both files, and return an audit summary."""
    manifest = load_manifest(manifest_path)
    file_hashes: dict[str, str] = {}
    for name, metadata in manifest["files"].items():
        path = raw_dir / name
        if not path.is_file():
            raise DataValidationError(f"missing raw file: {path}")
        actual = sha256_file(path)
        if actual != metadata["sha256"]:
            raise DataValidationError(
                f"checksum mismatch for {name}: expected {metadata['sha256']}, found {actual}"
            )
        file_hashes[name] = actual

    selections = load_selections(raw_dir / "c13k_selections.csv")
    if len(selections) != manifest["expected_selection_rows"]:
        raise DataValidationError("selection row count differs from acquisition manifest")
    unique_problems = len({record.problem for record in selections})
    if unique_problems != manifest["expected_unique_problem_ids"]:
        raise DataValidationError("unique Problem count differs from acquisition manifest")
    problems = load_problems(raw_dir / "c13k_problems.json", selections)
    summary = summarize_dataset(selections, problems, file_hashes)
    summary["dataset"] = manifest["dataset"]
    summary["source_repository"] = manifest["source_repository"]
    summary["source_commit"] = manifest["commit"]
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("choices13k validation: PASS")
    print(
        f"rows={summary['selection_rows']:,}; unique Problem IDs="
        f"{summary['unique_problem_ids']:,}; JSON rows={summary['json_rows']:,}"
    )
    print(
        f"missing={sum(summary['missing_values'].values())}; "
        f"exact duplicates={summary['exact_duplicate_rows']}; "
        f"bRate range=[{summary['bRate']['min']}, {summary['bRate']['max']}]"
    )
    counts = summary["sample_counts_n"]
    print(
        f"n: min={counts['min']}, median={counts['median']}, max={counts['max']}, "
        f"mean={counts['mean']:.3f}, sum across rows={counts['sum_across_rows']:,}"
    )
    print(f"problem row multiplicity={summary['problem_row_multiplicity']}")


def main() -> None:
    """Run the complete choices13k validation and optionally save its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = load_and_validate(args.raw_dir, args.manifest)
    _print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"summary={args.output}")


if __name__ == "__main__":
    main()

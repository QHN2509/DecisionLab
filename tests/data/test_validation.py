from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from decisionlab.data.validation import (
    CSV_COLUMNS,
    DataValidationError,
    load_problems,
    load_selections,
    summarize_dataset,
)


def valid_row(**overrides: str) -> dict[str, str]:
    row = {
        "Problem": "1",
        "Feedback": "False",
        "n": "15",
        "Block": "1",
        "Ha": "10",
        "pHa": "0.5",
        "La": "0",
        "Hb": "8",
        "pHb": "0.5",
        "Lb": "2",
        "LotShapeB": "0",
        "LotNumB": "1",
        "Amb": "False",
        "Corr": "0",
        "bRate": "0.6",
        "bRate_std": "0.2",
    }
    row.update(overrides)
    return row


def write_csv(
    path: Path, rows: list[dict[str, str]], columns: tuple[str, ...] = CSV_COLUMNS
) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_loader_parses_declared_types(tmp_path: Path) -> None:
    path = tmp_path / "selections.csv"
    write_csv(path, [valid_row()])

    record = load_selections(path)[0]

    assert record.problem == 1
    assert record.feedback is False
    assert record.n == 15
    assert record.brate == pytest.approx(0.6)


def test_loader_rejects_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "selections.csv"
    write_csv(path, [valid_row(bRate="")])

    with pytest.raises(DataValidationError, match="missing values"):
        load_selections(path)


def test_loader_rejects_schema_changes(tmp_path: Path) -> None:
    path = tmp_path / "selections.csv"
    row = valid_row()
    write_csv(path, [{name: row[name] for name in CSV_COLUMNS[:-1]}], columns=CSV_COLUMNS[:-1])

    with pytest.raises(DataValidationError, match="unexpected CSV schema"):
        load_selections(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"Problem": "one"}, "invalid feature type"),
        ({"Feedback": "false"}, "exactly 'True' or 'False'"),
        ({"pHa": "1.1"}, "probabilities must be"),
        ({"Block": "6"}, "Block must be"),
        ({"LotShapeB": "0", "LotNumB": "2"}, "undefined shape"),
        ({"Feedback": "True", "Block": "1"}, "inconsistent with Block"),
    ],
)
def test_loader_rejects_critical_schema_contract_violations(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    path = tmp_path / "selections.csv"
    write_csv(path, [valid_row(**overrides)])

    with pytest.raises(DataValidationError, match=message):
        load_selections(path)


@pytest.mark.parametrize("value", ["-0.01", "1.01", "nan"])
def test_loader_rejects_invalid_brate(tmp_path: Path, value: str) -> None:
    path = tmp_path / "selections.csv"
    write_csv(path, [valid_row(bRate=value)])

    with pytest.raises(DataValidationError, match="bRate|non-finite"):
        load_selections(path)


def test_loader_rejects_exact_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "selections.csv"
    row = valid_row()
    write_csv(path, [row, row])

    with pytest.raises(DataValidationError, match="exact duplicate"):
        load_selections(path)


def test_loader_allows_paired_feedback_rows_with_same_structure(tmp_path: Path) -> None:
    path = tmp_path / "selections.csv"
    write_csv(
        path,
        [
            valid_row(),
            valid_row(Feedback="True", Block="2", n="16", bRate="0.7", bRate_std="0.3"),
        ],
    )

    records = load_selections(path)

    assert len(records) == 2
    assert {record.feedback for record in records} == {False, True}


def test_loader_rejects_changed_structure_for_same_problem(tmp_path: Path) -> None:
    path = tmp_path / "selections.csv"
    write_csv(
        path,
        [valid_row(), valid_row(Feedback="True", Block="2", Ha="11")],
    )

    with pytest.raises(DataValidationError, match="changes structure"):
        load_selections(path)


def test_problem_json_is_keyed_by_csv_row_and_matches_expected_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "selections.csv"
    json_path = tmp_path / "problems.json"
    write_csv(csv_path, [valid_row()])
    json_path.write_text(
        json.dumps({"0": {"A": [[0.5, 10], [0.5, 0]], "B": [[0.5, 8], [0.5, 2]]}}),
        encoding="utf-8",
    )
    selections = load_selections(csv_path)

    problems = load_problems(json_path, selections)
    summary = summarize_dataset(selections, problems)

    assert summary["selection_rows"] == 1
    assert summary["unique_problem_ids"] == 1
    assert summary["missing_values"] == dict.fromkeys(CSV_COLUMNS, 0)


def test_problem_json_rejects_noncontiguous_row_keys(tmp_path: Path) -> None:
    csv_path = tmp_path / "selections.csv"
    json_path = tmp_path / "problems.json"
    write_csv(csv_path, [valid_row()])
    json_path.write_text(json.dumps({"1": {"A": [], "B": []}}), encoding="utf-8")

    with pytest.raises(DataValidationError, match="zero-based CSV row indexes"):
        load_problems(json_path, load_selections(csv_path))


def test_problem_json_rejects_changed_description_across_conditions(tmp_path: Path) -> None:
    csv_path = tmp_path / "selections.csv"
    json_path = tmp_path / "problems.json"
    write_csv(
        csv_path,
        [valid_row(), valid_row(Feedback="True", Block="2", n="16", bRate="0.7")],
    )
    json_path.write_text(
        json.dumps(
            {
                "0": {"A": [[0.5, 10], [0.5, 0]], "B": [[0.5, 8], [0.5, 2]]},
                "1": {"A": [[0.5, 10], [0.5, 0]], "B": [[0.5, 9], [0.5, 1]]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="changes across conditions"):
        load_problems(json_path, load_selections(csv_path))


@pytest.mark.parametrize(
    ("outcomes", "message"),
    [
        ([[0.4, 8], [0.5, 2]], "probabilities do not sum"),
        ([[1.0, "bad"]], "invalid B outcome"),
        ([[1.0, 5, 2]], r"must be \[probability, payoff\]"),
    ],
)
def test_problem_json_rejects_malformed_outcomes(
    tmp_path: Path, outcomes: list[list[object]], message: str
) -> None:
    csv_path = tmp_path / "selections.csv"
    json_path = tmp_path / "problems.json"
    write_csv(csv_path, [valid_row()])
    json_path.write_text(
        json.dumps({"0": {"A": [[0.5, 10], [0.5, 0]], "B": outcomes}}),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match=message):
        load_problems(json_path, load_selections(csv_path))

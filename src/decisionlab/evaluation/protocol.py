"""Reproduce historical development-era partitions; not canonical evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decisionlab import __version__
from decisionlab.data.fetch import DEFAULT_DESTINATION, DEFAULT_MANIFEST, sha256_file
from decisionlab.data.validation import (
    SelectionRecord,
    load_and_validate,
    load_problems,
    load_selections,
)
from decisionlab.evaluation.splitting import (
    SplitAssignment,
    audit_grouped_assignments,
    create_grouped_assignments,
    ordinary_row_split_leakage_demo,
    structural_fingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "historical_partitions.json"
DEFAULT_ASSIGNMENTS = PROJECT_ROOT / "data" / "processed" / "choices13k_splits.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "artifacts" / "manifests" / "split_summary.json"
DEFAULT_DOCUMENTATION = PROJECT_ROOT / "docs" / "historical_partitions.md"


def load_evaluation_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the historical partition contract retained only for provenance."""
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    required = {
        "split_seed",
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "primary_metric",
        "secondary_metrics",
        "weighted_sensitivity_metrics",
    }
    if set(config) != required:
        raise ValueError(f"Evaluation config must define exactly: {sorted(required)}")
    fractions = [
        config["train_fraction"],
        config["validation_fraction"],
        config["test_fraction"],
    ]
    if not all(isinstance(value, int | float) and value > 0.0 for value in fractions):
        raise ValueError("Evaluation fractions must be positive numbers")
    if not abs(sum(fractions) - 1.0) < 1e-12:
        raise ValueError("Evaluation fractions must sum to one")
    if config["primary_metric"] != "problem_group_equal_weighted_mae":
        raise ValueError("Historical primary metric must be equal-structural-group MAE")
    return config


def investigate_problem_feedback_block(
    selections: list[SelectionRecord], problems: dict[str, Any]
) -> dict[str, Any]:
    """Describe how condition rows relate to the underlying problem identity."""
    by_problem: dict[int, list[SelectionRecord]] = defaultdict(list)
    fingerprints_by_problem: dict[int, set[str]] = defaultdict(set)
    problems_by_fingerprint: dict[str, set[int]] = defaultdict(set)
    feedback_block = Counter()
    for row_index, record in enumerate(selections):
        fingerprint = structural_fingerprint(record, problems[str(row_index)])
        by_problem[record.problem].append(record)
        fingerprints_by_problem[record.problem].add(fingerprint)
        problems_by_fingerprint[fingerprint].add(record.problem)
        feedback_block[(str(record.feedback), record.block)] += 1

    multiplicity = Counter(len(rows) for rows in by_problem.values())
    paired = [rows for rows in by_problem.values() if len(rows) == 2]
    paired_condition_errors = sum(
        {row.feedback for row in rows} != {False, True} for rows in paired
    )
    return {
        "selection_rows": len(selections),
        "unique_problem_ids": len(by_problem),
        "problem_row_multiplicity": {
            str(size): count for size, count in sorted(multiplicity.items())
        },
        "paired_feedback_no_feedback_problem_ids": len(paired),
        "paired_condition_errors": paired_condition_errors,
        "feedback_block_counts": {
            f"feedback={feedback},block={block}": count
            for (feedback, block), count in sorted(feedback_block.items())
        },
        "no_feedback_outside_block_1": sum(
            not row.feedback and row.block != 1 for row in selections
        ),
        "feedback_in_block_1": sum(row.feedback and row.block == 1 for row in selections),
        "problem_ids_with_multiple_fingerprints": sum(
            len(values) > 1 for values in fingerprints_by_problem.values()
        ),
        "fingerprints_shared_by_problem_ids": sum(
            len(values) > 1 for values in problems_by_fingerprint.values()
        ),
        "unique_structural_fingerprints": len(problems_by_fingerprint),
    }


def summarize_partitions(
    selections: list[SelectionRecord], assignments: list[SplitAssignment]
) -> dict[str, Any]:
    """Summarize only structural and condition balance, without inspecting split targets."""
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        indices = [assignment.row_index for assignment in assignments if assignment.split == split]
        rows = [selections[index] for index in indices]
        result[split] = {
            "rows": len(rows),
            "problem_ids": len({row.problem for row in rows}),
            "structural_groups": len(
                {
                    assignment.structural_fingerprint
                    for assignment in assignments
                    if assignment.split == split
                }
            ),
            "feedback_rows": dict(sorted(Counter(str(row.feedback) for row in rows).items())),
            "ambiguity_rows": dict(sorted(Counter(str(row.amb) for row in rows).items())),
            "block_rows": {
                str(block): count
                for block, count in sorted(Counter(row.block for row in rows).items())
            },
        }
    return result


def _write_assignments(path: Path, assignments: list[SplitAssignment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    columns = ["row_index", "problem", "structural_fingerprint", "split"]
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=columns)
            writer.writeheader()
            writer.writerows(asdict(assignment) for assignment in assignments)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_evaluation_protocol(summary: dict[str, Any], path: Path) -> None:
    """Generate the evaluation protocol from the executed grouping investigation."""
    relationships = summary["problem_feedback_block_relationship"]
    grouped = summary["grouped_split_audit"]
    row_demo = summary["ordinary_row_split_demo"]
    partitions = summary["partition_summary"]
    config = summary["config"]
    lines = [
        "# Historical development-era partitions",
        "",
        (
            "> **Not canonical evaluation.** Target-aware EDA preceded these partitions. "
            "They are not an untouched or confirmatory holdout and are superseded by nested "
            "grouped cross-validation."
        ),
        "",
        (
            "This protocol was generated before model training from the checksum-validated "
            f"choices13k data at commit `{summary['source_commit']}`. It locks grouping, "
            "partitioning, metrics, weighting, and test-use rules."
        ),
        "",
        "## Problem, feedback, and Block investigation",
        "",
        (
            f"The CSV has {relationships['selection_rows']:,} rows and "
            f"{relationships['unique_problem_ids']:,} unique `Problem` IDs. "
            f"{relationships['problem_row_multiplicity']['1']:,} problems occur once and "
            f"{relationships['problem_row_multiplicity']['2']:,} occur twice. Every repeated "
            "problem is one no-feedback row plus one feedback row with identical gamble structure."
        ),
        "",
        "Observed `Feedback`/`Block` counts:",
        "",
        "| Feedback | Block | Rows |",
        "|---|---:|---:|",
    ]
    for key, count in relationships["feedback_block_counts"].items():
        feedback, block = key.replace("feedback=", "").replace("block=", "").split(",")
        lines.append(f"| {feedback} | {block} | {count:,} |")
    lines.extend(
        [
            "",
            (
                "No-feedback observations are always in `Block=1`; feedback observations are "
                "always in blocks 2–5. Therefore, `Block` is not an underlying-problem ID and "
                "is a deterministic encoding of feedback status in this release. `Feedback` is "
                "a condition attached to a problem, not the split group."
            ),
            "",
            "## Grouping variable",
            "",
            (
                "The split group is `structural_fingerprint`: SHA-256 over both full gamble "
                "distributions plus `Ha`, `pHa`, `La`, `Hb`, `pHb`, `Lb`, `LotShapeB`, "
                "`LotNumB`, `Amb`, and `Corr`. It deliberately excludes `Feedback`, `Block`, "
                "`n`, `bRate`, and `bRate_std`."
            ),
            "",
            (
                "At the pinned source revision, "
                f"{relationships['unique_structural_fingerprints']:,} "
                "fingerprints map one-to-one to the same number of `Problem` IDs: there are "
                f"{relationships['problem_ids_with_multiple_fingerprints']} Problem IDs with "
                "multiple fingerprints and "
                f"{relationships['fingerprints_shared_by_problem_ids']} fingerprints shared by "
                "different Problem IDs. Thus, grouping by `Problem` is sufficient for the current "
                "release, while the fingerprint is the more defensive implementation because it "
                "would also merge exact structures assigned different IDs."
            ),
            "",
            "## Why ordinary row splitting leaks",
            "",
            (
                "A row-level split treats feedback variants as independent. A learner can then see "
                "the exact gamble structure under one feedback condition during training and be "
                "evaluated on the same structure under another condition. That makes prediction "
                "artificially easy even when `Feedback` is present as a feature."
            ),
            "",
            (
                "The deterministic row-split demonstration placed "
                f"{row_demo['structural_groups_crossing_splits']:,} "
                f"of {row_demo['repeated_structural_groups']:,} repeated structural groups across "
                "multiple partitions. Singleton problems cannot create this exact paired-row leak, "
                "but grouping remains required because the dataset contains paired problems and "
                "future revisions could contain duplicate structures under different IDs."
            ),
            "",
            "## Locked grouped partitions",
            "",
            (
                f"Groups are assigned by stable SHA-256 hashing with seed `{config['split_seed']}` "
                f"and thresholds {config['train_fraction']:.0%}/"
                f"{config['validation_fraction']:.0%}/{config['test_fraction']:.0%}. "
                "The hash uses no target values, and adding unrelated rows does not reshuffle "
                "existing groups."
            ),
            "",
            "| Split | Rows | Structural groups | Feedback rows | No-feedback rows |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "validation", "test"):
        values = partitions[split]
        lines.append(
            f"| {split} | {values['rows']:,} | {values['structural_groups']:,} | "
            f"{values['feedback_rows'].get('True', 0):,} | "
            f"{values['feedback_rows'].get('False', 0):,} |"
        )
    lines.extend(
        [
            "",
            (
                f"Audit result: **{grouped['status']}** with "
                f"{grouped['structural_group_overlap_count']} structural-group overlaps and "
                f"{grouped['problem_id_overlap_count']} Problem-ID overlaps."
            ),
            "",
            "The validation split is used for model and hyperparameter selection. The test split "
            "must remain untouched until a final pipeline is selected. Any later cross-validation "
            "must also group by structural fingerprint and remain inside the development data.",
            "",
            (
                "The earlier EDA inspected the full dataset before these assignments existed. "
                "Consequently, hypotheses recorded by that EDA can be treated as prespecified for "
                "future modeling, but the final test set is not untouched with respect to broad "
                "exploratory knowledge. New test-set patterns must not be presented as "
                "confirmatory."
            ),
            "",
            "## Metrics for bRate",
            "",
            "For structural problem groups `g=1,…,G`, let each group contain condition rows `i`.",
            "Predictions must be finite and in `[0,1]`; evaluation never clips them silently.",
            "",
            "### Primary equal-structural-group metric",
            "",
            (
                "- **Problem-group MAE:** first compute mean absolute error within each structural "
                "problem group, then average those group losses equally. This is the locked "
                "model-selection metric, so paired condition rows and singleton problems receive "
                "the same total primary weight."
            ),
            "",
            "### Secondary condition-row metrics",
            "",
            (
                "- **Condition-row MAE/RMSE:** ordinary row-level summaries retained for "
                "comparison with earlier runs; paired problems contribute once per condition row."
            ),
            (
                "- **Condition-row R² and mean bias:** secondary diagnostics computed over "
                "condition rows; R² may be negative."
            ),
            (
                "- **Calibration diagnostics:** binned observed-versus-predicted plots plus "
                "calibration intercept and slope. These are diagnostics, not tuning targets "
                "on the test set."
            ),
            "",
            "### Participant-count-weighted sensitivity metrics",
            "",
            (
                "Repeat MAE, RMSE, R², and mean bias using normalized weights "
                "`wᵢ = nᵢ / Σnᵢ`. Larger-`n` rows estimate aggregate rates more precisely, "
                "so these "
                "metrics are informative sensitivity analyses. They answer a different question—"
                "performance weighted by participant-row contributions—and must not replace the "
                "primary problem-level metric."
            ),
            "",
            (
                "Do not interpret `5n` repeated responses as independent Bernoulli trials: each "
                "participant contributed five responses to a problem. For that reason, binomial "
                "log loss or likelihood weighted by `5n` is not a primary metric without a model "
                "that accounts for within-participant clustering."
            ),
            "",
            "## Comparison and uncertainty rules",
            "",
            "- Compare candidate pipelines on identical validation rows and the same primary MAE.",
            "- Report all prespecified metrics; do not select whichever metric favors a model.",
            (
                "- Estimate uncertainty and pairwise model differences by resampling structural "
                "groups, keeping all feedback variants together within each bootstrap replicate."
            ),
            (
                "- Report overall metrics first. Feedback, ambiguity, correlation, and "
                "lottery-shape subgroup metrics are prespecified diagnostics and require "
                "row/group counts."
            ),
            (
                "- Every reported value must be generated from saved predictions and the locked "
                "split assignments. No model results are produced by this protocol stage."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_evaluation_protocol(
    raw_dir: Path = DEFAULT_DESTINATION,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    assignments_path: Path = DEFAULT_ASSIGNMENTS,
    summary_path: Path = DEFAULT_SUMMARY,
    documentation_path: Path = DEFAULT_DOCUMENTATION,
) -> dict[str, Any]:
    """Validate data, lock grouped assignments, and persist the protocol audit."""
    validation = load_and_validate(raw_dir, manifest_path)
    selections = load_selections(raw_dir / "c13k_selections.csv")
    problems = load_problems(raw_dir / "c13k_problems.json", selections)
    config = load_evaluation_config(config_path)
    relationships = investigate_problem_feedback_block(selections, problems)
    assignments = create_grouped_assignments(
        selections,
        problems,
        seed=config["split_seed"],
        train_fraction=config["train_fraction"],
        validation_fraction=config["validation_fraction"],
        test_fraction=config["test_fraction"],
    )
    grouped_audit = audit_grouped_assignments(assignments)
    row_demo = ordinary_row_split_leakage_demo(
        selections,
        problems,
        seed=config["split_seed"],
        train_fraction=config["train_fraction"],
        validation_fraction=config["validation_fraction"],
    )
    partition_summary = summarize_partitions(selections, assignments)
    _write_assignments(assignments_path, assignments)
    summary: dict[str, Any] = {
        "dataset": validation["dataset"],
        "source_commit": validation["source_commit"],
        "source_sha256": validation["sha256"],
        "decisionlab_version": __version__,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "problem_feedback_block_relationship": relationships,
        "grouped_split_audit": grouped_audit,
        "ordinary_row_split_demo": row_demo,
        "partition_summary": partition_summary,
        "assignments_output": {
            "path": str(assignments_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(assignments_path),
        },
        "target_statistics_by_split_inspected": False,
        "models_trained": False,
    }
    write_evaluation_protocol(summary, documentation_path)
    summary["documentation_output"] = {
        "path": str(documentation_path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(documentation_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Create the locked grouped splits and evaluation documentation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--documentation", type=Path, default=DEFAULT_DOCUMENTATION)
    args = parser.parse_args()
    summary = build_evaluation_protocol(
        args.raw_dir,
        args.manifest,
        args.config,
        args.assignments,
        args.summary,
        args.documentation,
    )
    audit = summary["grouped_split_audit"]
    demo = summary["ordinary_row_split_demo"]
    print(
        f"Grouped split: {audit['status']}; rows={sum(audit['rows'].values()):,}; "
        f"groups={audit['unique_structural_groups']:,}; overlap="
        f"{audit['structural_group_overlap_count']}; ordinary-row overlap="
        f"{demo['structural_groups_crossing_splits']:,}"
    )


if __name__ == "__main__":
    main()

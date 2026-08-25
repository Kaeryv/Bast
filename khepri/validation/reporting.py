"""Machine-first serialization and small human-readable reports."""

from datetime import datetime, timezone
import json
from pathlib import Path
from platform import platform, python_version
from typing import Optional

import numpy as np

from .models import CaseRun, ValidationCase


def _comparisons(case: ValidationCase, run: CaseRun) -> list:
    comparisons = []
    for evaluation in run.evaluations:
        reference = case.reference(evaluation.configuration)
        if reference is None:
            comparisons.append(None)
            continue
        comparisons.append(
            {
                "configuration_key": evaluation.configuration.key,
                "reference_kind": reference.kind,
                "reference_observables": dict(reference.observables),
                "maximum_absolute_error": case.error(evaluation, reference),
                "fixture": reference.fixture,
                "uncertainty": reference.uncertainty,
            }
        )
    return comparisons


def write_json_run(case: ValidationCase, run: CaseRun, output) -> Path:
    """Write the lossless machine-readable artifact used to build reports."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = run.to_dict()
    aggregate = case.collect(run.evaluations)
    aggregate_reference = case.aggregate_reference(aggregate)
    aggregate_comparison = None
    if aggregate_reference is not None:
        aggregate_comparison = {
            "reference_kind": aggregate_reference.kind,
            "reference_observables": dict(aggregate_reference.observables),
            "maximum_absolute_error": case.aggregate_error(
                aggregate, aggregate_reference
            ),
            "fixture": aggregate_reference.fixture,
            "uncertainty": aggregate_reference.uncertainty,
        }
    document.update(
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python": python_version(),
                "numpy": np.__version__,
                "platform": platform(),
            },
            "case": {
                "title": case.title,
                "description": case.description,
                "tags": list(case.tags),
                "references": [
                    {
                        "citation": reference.citation,
                        "url": reference.url,
                        "note": reference.note,
                    }
                    for reference in case.references
                ],
            },
            "aggregate": aggregate.to_dict(),
            "aggregate_comparison": aggregate_comparison,
            "comparisons": _comparisons(case, run),
        }
    )
    output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return output


def write_markdown_run(
    case: ValidationCase,
    run: CaseRun,
    output,
    raw_json: Optional[Path] = None,
) -> Path:
    """Write a concise report derived from an already completed run."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregate = case.collect(run.evaluations)
    aggregate_reference = case.aggregate_reference(aggregate)
    lines = [
        f"# {case.title}",
        "",
        case.description,
        "",
        f"Runner: `{run.runner}` with {run.workers} worker(s).",
        "",
        f"Configurations: {len(run.evaluations)}; process wall time: {run.wall_seconds:.6f} s.",
        "",
    ]
    if raw_json is not None:
        lines.extend([f"Machine-readable source: `{Path(raw_json).name}`.", ""])
    if case.references:
        lines.extend(["## References", ""])
        for reference in case.references:
            note = f" — {reference.note}" if reference.note else ""
            lines.append(f"- [{reference.citation}]({reference.url}){note}")
        lines.append("")
    lines.extend(["## Aggregate observables", ""])
    if aggregate.observables:
        lines.extend(["| Observable | Value |", "|---|---:|"])
        for name, value in aggregate.observables.items():
            lines.append(f"| {name} | {value:.12g} |")
    else:
        lines.append("No scalar aggregate observables.")
    if aggregate_reference is not None:
        lines.extend(
            [
                "",
                f"Aggregate reference basis: `{aggregate_reference.kind}`; "
                f"maximum absolute error: "
                f"`{case.aggregate_error(aggregate, aggregate_reference):.6g}`.",
            ]
        )
    lines.extend(["", "## Configuration results", ""])
    observable_names = sorted(
        {
            name
            for evaluation in run.evaluations
            for name in evaluation.result.observables
        }
    )
    comparisons = _comparisons(case, run)
    lines.append(
        "| # | Configuration | Wall [s] | Reference error | "
        + " | ".join(observable_names)
        + " |"
    )
    lines.append("|---:|---|---:|---:|" + "---:|" * len(observable_names))
    for evaluation, comparison in zip(run.evaluations, comparisons):
        values = " | ".join(
            f"{evaluation.result.observables.get(name, float('nan')):.12g}"
            for name in observable_names
        )
        reference_error = (
            "—"
            if comparison is None
            else f"{comparison['maximum_absolute_error']:.6g}"
        )
        lines.append(
            f"| {evaluation.configuration.ordinal} | `{evaluation.configuration.key}` | "
            f"{evaluation.wall_seconds:.6f} | {reference_error} | {values} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output

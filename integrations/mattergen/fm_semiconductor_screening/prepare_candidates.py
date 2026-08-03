#!/usr/bin/env python3
"""Export MatterSim-screened structures as CIF files and a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--stability-threshold", type=float, default=0.10)
    return parser.parse_args()


def load_metric_lists(path: Path) -> dict[str, list[Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {key: value for key, value in data.items() if isinstance(value, list)}


def metric_value(
    metrics: dict[str, list[Any]], index: int, names: tuple[str, ...], default: Any
) -> Any:
    for name in names:
        values = metrics.get(name)
        if values is not None and index < len(values):
            return values[index]
    return default


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def space_group(atoms: Any) -> str:
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = AseAtomsAdaptor.get_structure(atoms)
        return SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_symbol()
    except Exception:
        return "P1"


def main() -> None:
    import ase.io

    args = parse_args()
    results_dir = args.results_dir.resolve()
    metrics_path = results_dir / "detailed_metrics.json"
    structures_path = results_dir / "relaxed.extxyz"
    output_dir = results_dir / "stable_cifs"
    output_csv = results_dir / "mattergen_screening.csv"
    stable_csv = results_dir / "stable_candidates.csv"

    for path in (metrics_path, structures_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required workflow output is missing: {path}")

    metrics = load_metric_lists(metrics_path)
    structures = ase.io.read(structures_path, index=":")
    if not isinstance(structures, list):
        structures = [structures]

    metric_lengths = {len(values) for values in metrics.values() if values}
    if metric_lengths and len(structures) not in metric_lengths:
        raise ValueError(
            f"Structure count ({len(structures)}) does not match metric lengths "
            f"({sorted(metric_lengths)})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "cif_file",
        "formula",
        "space_group",
        "energy_above_hull_eV_atom",
        "stable",
        "unique",
        "novel",
        "structure_valid",
        "composition_valid",
        "passed_mattersim_screen",
        "screen_reason",
        "status",
    ]

    rows: list[dict[str, Any]] = []
    for index, atoms in enumerate(structures):
        candidate_id = f"gen_{index}"
        e_hull = finite_float(
            metric_value(
                metrics,
                index,
                ("energy_above_hull", "energy_above_hull_per_atom"),
                math.nan,
            )
        )
        stable_from_metric = metric_value(metrics, index, ("stable",), None)
        stable = (
            as_bool(stable_from_metric)
            if stable_from_metric is not None
            else math.isfinite(e_hull) and e_hull <= args.stability_threshold
        )
        unique = as_bool(metric_value(metrics, index, ("unique", "is_unique"), True), True)
        novel = as_bool(metric_value(metrics, index, ("novel", "is_novel"), False))
        structure_valid = as_bool(
            metric_value(metrics, index, ("structure_validity",), True), True
        )
        composition_valid = as_bool(
            metric_value(metrics, index, ("comp_validity",), True), True
        )

        reasons: list[str] = []
        if not math.isfinite(e_hull):
            reasons.append("missing_energy_above_hull")
        elif e_hull > args.stability_threshold:
            reasons.append("energy_above_hull_above_threshold")
        if not stable:
            reasons.append("not_stable")
        if not unique:
            reasons.append("duplicate")
        if not structure_valid:
            reasons.append("invalid_structure")
        if not composition_valid:
            reasons.append("invalid_composition")

        passed = not reasons
        cif_relative = ""
        if passed:
            cif_name = f"{candidate_id}.cif"
            cif_path = output_dir / cif_name
            ase.io.write(cif_path, atoms, format="cif")
            cif_relative = cif_path.relative_to(results_dir).as_posix()

        rows.append(
            {
                "candidate_id": candidate_id,
                "cif_file": cif_relative,
                "formula": atoms.get_chemical_formula(mode="hill"),
                "space_group": space_group(atoms),
                "energy_above_hull_eV_atom": "" if not math.isfinite(e_hull) else e_hull,
                "stable": stable,
                "unique": unique,
                "novel": novel,
                "structure_valid": structure_valid,
                "composition_valid": composition_valid,
                "passed_mattersim_screen": passed,
                "screen_reason": "passed" if passed else ";".join(dict.fromkeys(reasons)),
                "status": "MatterSim-screened, not DFT-validated",
            }
        )

    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    passed_rows = [row for row in rows if as_bool(row["passed_mattersim_screen"])]
    passed_rows.sort(
        key=lambda row: (
            0 if as_bool(row["novel"]) else 1,
            float(row["energy_above_hull_eV_atom"]),
            row["candidate_id"],
        )
    )
    with stable_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(passed_rows)

    passed_count = len(passed_rows)
    print(f"Wrote {output_csv}")
    print(f"Wrote {stable_csv}")
    print(f"MatterSim screen: {passed_count}/{len(rows)} structures passed")
    print(f"Exported CIF files to {output_dir}")


if __name__ == "__main__":
    main()

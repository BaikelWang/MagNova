#!/usr/bin/env python3
"""Dependency-free unit tests for the workflow's parsing and ranking logic."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, WORKFLOW_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_candidates", "prepare_candidates.py")


class PrepareCandidatesTests(unittest.TestCase):
    def test_metric_alias_and_defaults(self) -> None:
        metrics = {"energy_above_hull_per_atom": [0.04], "unique": [True]}
        self.assertEqual(
            prepare.metric_value(
                metrics,
                0,
                ("energy_above_hull", "energy_above_hull_per_atom"),
                None,
            ),
            0.04,
        )
        self.assertEqual(prepare.metric_value(metrics, 0, ("novel",), False), False)

    def test_load_metric_lists_ignores_non_lists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(
                json.dumps({"stable": [True, False], "metadata": {"version": 1}}),
                encoding="utf-8",
            )
            self.assertEqual(prepare.load_metric_lists(path), {"stable": [True, False]})

    def test_finite_float_rejects_nan(self) -> None:
        self.assertTrue(prepare.finite_float("0.08") == 0.08)
        self.assertTrue(prepare.finite_float("nan") != prepare.finite_float("nan"))


if __name__ == "__main__":
    unittest.main()

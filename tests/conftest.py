"""Public-test configuration.

Dataset-builder tests remain available when the private SFT corpora are mounted
under ``data/``. They are not collected in the source-only public release.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if not (ROOT / "data" / "sft_v1" / "train.jsonl").exists():
    collect_ignore_glob = ["test_sft*_dataset.py"]

from __future__ import annotations

import argparse

from .agent import MagNovaAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MagNova material requirement agent MVP")
    parser.add_argument("text", nargs="+", help="natural-language material request")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outcome = MagNovaAgent().run(" ".join(args.text))
    print(outcome.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("minimind-3"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/magnova_base_768.pth"),
    )
    args = parser.parse_args()

    if args.output.exists():
        print(f"already_exists path={args.output} sha256={sha256(args.output)}")
        return

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.source,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    state_dict = {
        key: value.detach().cpu().half()
        for key, value in hf_model.state_dict().items()
    }

    native_model = MiniMindForCausalLM(
        MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    )
    missing, unexpected = native_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"incompatible state_dict missing={missing} unexpected={unexpected}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(state_dict, temporary)
    temporary.replace(args.output)
    print(
        f"saved path={args.output} tensors={len(state_dict)} "
        f"sha256={sha256(args.output)}"
    )


if __name__ == "__main__":
    main()

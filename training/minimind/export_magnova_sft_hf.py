from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from model.model_minimind import MiniMindConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight",
        type=Path,
        default=Path("out/magnova_sft_v1_768.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("magnova-sft-v1"),
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("model"))
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("out/magnova_sft_v1_metrics.json"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/sft_v1/manifest.json"),
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    mini_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    qwen_config = Qwen3Config(
        vocab_size=mini_config.vocab_size,
        hidden_size=mini_config.hidden_size,
        intermediate_size=mini_config.intermediate_size,
        num_hidden_layers=mini_config.num_hidden_layers,
        num_attention_heads=mini_config.num_attention_heads,
        num_key_value_heads=mini_config.num_key_value_heads,
        head_dim=mini_config.head_dim,
        max_position_embeddings=mini_config.max_position_embeddings,
        rms_norm_eps=mini_config.rms_norm_eps,
        rope_theta=mini_config.rope_theta,
        tie_word_embeddings=mini_config.tie_word_embeddings,
        use_sliding_window=False,
        sliding_window=None,
    )
    model = Qwen3ForCausalLM(qwen_config)
    state_dict = torch.load(args.weight, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.half().save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.save_pretrained(args.output)

    metadata = {
        "source_weight": str(args.weight),
        "training_metrics": json.loads(args.metrics.read_text(encoding="utf-8")),
        "dataset_manifest": json.loads(
            args.dataset_manifest.read_text(encoding="utf-8")
        ),
    }
    (args.output / "magnova_training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    loaded = AutoModelForCausalLM.from_pretrained(args.output)
    if sum(parameter.numel() for parameter in loaded.parameters()) != sum(
        parameter.numel() for parameter in model.parameters()
    ):
        raise RuntimeError("exported parameter count mismatch")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "parameters": sum(
                    parameter.numel() for parameter in loaded.parameters()
                ),
                "files": sorted(path.name for path in args.output.iterdir()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total_steps: int, peak_lr: float) -> float:
    progress = min(step / max(total_steps, 1), 1.0)
    return peak_lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(
    model: MiniMindForCausalLM,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for input_ids, labels in loader:
            result = model(
                input_ids.to(device, non_blocking=True),
                labels=labels.to(device, non_blocking=True),
            )
            losses.append(float(result.loss.item()))
    model.train()
    return sum(losses) / len(losses)


def save_state_dict(model: MiniMindForCausalLM, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state_dict = {
        key: value.detach().cpu().half()
        for key, value in model.state_dict().items()
    }
    torch.save(state_dict, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--dev-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("model"))
    parser.add_argument(
        "--base-weight",
        type=Path,
        default=Path("out/magnova_base_768.pth"),
    )
    parser.add_argument(
        "--output-weight",
        type=Path,
        default=Path("out/magnova_sft_v1_768.pth"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("out/magnova_sft_v1_metrics.json"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run")
    set_seed(args.seed)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_dataset = SFTDataset(
        str(args.train_data),
        tokenizer,
        max_length=args.max_seq_len,
    )
    dev_dataset = SFTDataset(
        str(args.dev_data),
        tokenizer,
        max_length=args.max_seq_len,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    model = MiniMindForCausalLM(config)
    state_dict = torch.load(args.base_weight, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    updates_per_epoch = math.ceil(
        len(train_loader) / args.accumulation_steps
    )
    total_updates = updates_per_epoch * args.epochs
    global_update = 0
    best_dev_loss = float("inf")
    history: list[dict] = []

    baseline_dev_loss = evaluate(model, dev_loader, device)
    print(
        json.dumps(
            {"event": "baseline", "dev_loss": baseline_dev_loss},
            ensure_ascii=False,
        ),
        flush=True,
    )
    optimizer.zero_grad(set_to_none=True)
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        running_loss = 0.0
        batches_since_log = 0
        for batch_index, (input_ids, labels) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = model(input_ids, labels=labels)
                raw_loss = result.loss
                loss = raw_loss / args.accumulation_steps
            loss.backward()
            running_loss += float(raw_loss.item())
            batches_since_log += 1

            should_update = (
                batch_index % args.accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if not should_update:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            global_update += 1
            learning_rate = cosine_lr(
                global_update,
                total_updates,
                args.learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if (
                global_update % args.log_interval == 0
                or batch_index == len(train_loader)
            ):
                print(
                    json.dumps(
                        {
                            "event": "train",
                            "epoch": epoch,
                            "update": global_update,
                            "train_loss": running_loss / batches_since_log,
                            "learning_rate": learning_rate,
                            "gpu_memory_gb": round(
                                torch.cuda.max_memory_allocated() / 1024**3,
                                3,
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                running_loss = 0.0
                batches_since_log = 0

        dev_loss = evaluate(model, dev_loader, device)
        epoch_record = {
            "epoch": epoch,
            "dev_loss": dev_loss,
            "epoch_seconds": round(time.perf_counter() - epoch_started, 3),
        }
        history.append(epoch_record)
        print(
            json.dumps(
                {"event": "epoch_end", **epoch_record},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_state_dict(model, args.output_weight)
            print(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "path": str(args.output_weight),
                        "best_dev_loss": best_dev_loss,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    metrics = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
            "effective_batch_size": args.batch_size * args.accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "train_samples": len(train_dataset),
            "dev_samples": len(dev_dataset),
            "test_used_during_training": False,
        },
        "baseline_dev_loss": baseline_dev_loss,
        "best_dev_loss": best_dev_loss,
        "history": history,
        "total_seconds": round(time.perf_counter() - training_started, 3),
        "peak_gpu_memory_gb": round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3,
        ),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect how the MiniMind tokenizer converts text to tokens and IDs."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_SAMPLES = [
    "你好，MiniMind！",
    "请帮我设计一种高磁化强度、低重稀土依赖的材料。",
    "MatterGen should generate Fe-Co-B candidates with max_atoms=20.",
    "Fe-Co-B、Nd2Fe14B、磁矩密度 0.2 μB/Å³",
    '<tool_call>{"name":"submit_magnova_task"}</tool_call>',
]


def inspect_text(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    offsets = encoded["offset_mapping"]
    pieces = [
        {
            "source": text[start:end],
            "token": token,
            "id": token_id,
            "offset": [start, end],
        }
        for token_id, token, (start, end) in zip(ids, tokens, offsets)
    ]
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    return {
        "text": text,
        "char_count": len(text),
        "token_count": len(ids),
        "pieces": pieces,
        "decoded": decoded,
        "roundtrip_equal": decoded == text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./minimind-3")
    parser.add_argument(
        "--text",
        action="append",
        help="Text to inspect. Repeat this option for multiple samples.",
    )
    parser.add_argument("--show-chat", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    texts = args.text or DEFAULT_SAMPLES

    result = {
        "model_dir": str(model_dir.resolve()),
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "len_tokenizer": len(tokenizer),
            "model_max_length": tokenizer.model_max_length,
            "special_tokens_map": tokenizer.special_tokens_map,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
        },
        "samples": [inspect_text(tokenizer, text) for text in texts],
    }

    if args.show_chat:
        messages = [
            {"role": "system", "content": "你是 MagNova 材料设计助手。"},
            {
                "role": "user",
                "content": "请设计一种高磁性、低重稀土依赖的材料。",
            },
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        chat_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        result["chat"] = {
            "rendered": rendered,
            "token_count": len(chat_ids),
            "ids": chat_ids,
            "tokens": tokenizer.convert_ids_to_tokens(chat_ids),
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
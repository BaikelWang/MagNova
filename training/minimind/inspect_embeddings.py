#!/usr/bin/env python3
"""Inspect MiniMind token embeddings without loading the full model."""

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer


MODEL_DIR = Path("./minimind-3")
WEIGHT_FILE = MODEL_DIR / "model.safetensors"
EMBED_KEY = "model.embed_tokens.weight"
TERMS = [
    "材料",
    "设计",
    "结构",
    "候选",
    "生成",
    "磁",
    "性",
    "高",
    "低",
    "用户",
    "助手",
    "MatterGen",
]
PAIR_NAMES = [
    ("材料", "设计"),
    ("材料", "结构"),
    ("材料", "候选"),
    ("磁", "性"),
    ("高", "低"),
    ("用户", "助手"),
]


def cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator)


def readable_piece(tokenizer, token_id):
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    if not text or "�" in text or text.isspace():
        return None
    return text


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    with open(MODEL_DIR / "config.json", encoding="utf-8") as file:
        config = json.load(file)

    with safe_open(WEIGHT_FILE, framework="np") as file:
        keys = list(file.keys())
        embedding = file.get_tensor(EMBED_KEY).astype(np.float32)
        total_parameters = sum(int(np.prod(file.get_tensor(key).shape)) for key in keys)

    print("=== Embedding 矩阵 ===")
    print(f"shape: {embedding.shape}")
    print(f"stored_dtype: float16")
    print(f"parameters: {embedding.size:,}")
    print(f"stored_bytes: {embedding.size * 2:,}")
    print(f"total_model_parameters: {total_parameters:,}")
    print(f"parameter_share: {embedding.size / total_parameters:.2%}")
    print(f"tie_word_embeddings: {config['tie_word_embeddings']}")
    print(f"separate_lm_head_present: {'lm_head.weight' in keys}")

    term_info = {}
    print("\n=== 文本怎样变成向量 ===")
    for term in TERMS:
        ids = tokenizer(term, add_special_tokens=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"{term}: ids={ids}, tokens={tokens}")
        if len(ids) == 1:
            vector = embedding[ids[0]]
            term_info[term] = (ids[0], vector)
            print(
                "  single-token vector: "
                f"shape={vector.shape}, norm={np.linalg.norm(vector):.6f}, "
                f"mean={vector.mean():.6f}, std={vector.std():.6f}"
            )
            print(f"  first_8={vector[:8].round(6).tolist()}")
        else:
            sequence = embedding[ids]
            print(f"  multi-token embedding sequence shape={sequence.shape}")

    sentence = "请设计高磁性材料"
    sentence_ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
    print("\n=== 句子张量形状 ===")
    print(f"text: {sentence}")
    print(f"ids: {sentence_ids}")
    print(f"tokens: {tokenizer.convert_ids_to_tokens(sentence_ids)}")
    print(f"without_batch: {embedding[sentence_ids].shape}")
    print(f"with_batch: {(1, len(sentence_ids), embedding.shape[1])}")

    print("\n=== 单 Token 向量余弦相似度 ===")
    for left_name, right_name in PAIR_NAMES:
        if left_name not in term_info or right_name not in term_info:
            print(f"{left_name} vs {right_name}: skipped (至少一项不是单 Token)")
            continue
        value = cosine(term_info[left_name][1], term_info[right_name][1])
        print(f"{left_name} vs {right_name}: {value:.6f}")

    normalized = embedding / np.maximum(
        np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12
    )
    print("\n=== Embedding 最近邻（仅作观察）===")
    for name in ["材料", "设计", "磁", "高", "低"]:
        if name not in term_info:
            continue
        token_id, vector = term_info[name]
        scores = normalized @ (vector / max(np.linalg.norm(vector), 1e-12))
        order = np.argsort(-scores)
        neighbors = []
        for candidate_id in order:
            if int(candidate_id) in {0, 1, 2, int(token_id)}:
                continue
            decoded = readable_piece(tokenizer, candidate_id)
            if decoded is None:
                continue
            neighbors.append((decoded, int(candidate_id), float(scores[candidate_id])))
            if len(neighbors) == 6:
                break
        formatted = ", ".join(
            f"{text!r}(id={token_id}, cos={score:.4f})"
            for text, token_id, score in neighbors
        )
        print(f"{name}: {formatted}")

    print("\n注意：静态 Embedding 最近不等于上下文语义最近。")
    print("Token 经过 Transformer 后的隐藏状态才包含当前句子的上下文。")


if __name__ == "__main__":
    main()
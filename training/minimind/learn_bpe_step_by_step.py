#!/usr/bin/env python3
"""A tiny character-level BPE demonstration for learning purposes.

Real MiniMind uses a byte-level tokenizer. This script starts from Unicode
characters so every merge remains easy to read by hand.
"""

from collections import Counter


CORPUS = [
    ("磁性材料", 5),
    ("高磁性材料", 3),
    ("低成本材料", 2),
    ("材料设计", 2),
    ("材料生成", 2),
]


def count_pairs(segmented_corpus):
    counts = Counter()
    for tokens, frequency in segmented_corpus:
        for left, right in zip(tokens, tokens[1:]):
            counts[(left, right)] += frequency
    return counts


def merge_pair(tokens, pair):
    merged = []
    index = 0
    while index < len(tokens):
        if (
            index + 1 < len(tokens)
            and tokens[index] == pair[0]
            and tokens[index + 1] == pair[1]
        ):
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return merged


def apply_merges(text, merges):
    tokens = list(text)
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens


def print_corpus(segmented_corpus):
    for tokens, frequency in segmented_corpus:
        print(f"  {frequency} × {' | '.join(tokens)}")


def main():
    segmented = [(list(text), frequency) for text, frequency in CORPUS]
    merges = []

    print("=== 原始语料：从单个字符开始 ===")
    print_corpus(segmented)

    for step in range(1, 9):
        pair_counts = count_pairs(segmented)
        if not pair_counts:
            break
        best_pair, count = pair_counts.most_common(1)[0]
        merges.append(best_pair)
        segmented = [
            (merge_pair(tokens, best_pair), frequency)
            for tokens, frequency in segmented
        ]
        print(f"\n=== 第 {step} 次合并 ===")
        print(f"最高频相邻对：{best_pair[0]} + {best_pair[1]}，加权出现 {count} 次")
        print(f"新增 Token：{best_pair[0] + best_pair[1]}")
        print_corpus(segmented)

    print("\n=== 学到的合并规则（顺序很重要）===")
    for index, pair in enumerate(merges, start=1):
        print(f"  {index}. {pair[0]} + {pair[1]} -> {pair[0] + pair[1]}")

    print("\n=== 用学习到的规则切分新文本 ===")
    for text in ["高磁性材料设计", "低成本磁性材料", "新型材料生成"]:
        tokens = apply_merges(text, merges)
        print(f"  {text} -> {' | '.join(tokens)}  ({len(tokens)} tokens)")


if __name__ == "__main__":
    main()
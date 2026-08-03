#!/usr/bin/env python3
"""Compare two byte-level BPE tokenizers trained on different corpora."""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


GENERAL_SENTENCES = [
    "今天天气很好，我们一起学习人工智能。",
    "用户提出问题，助手给出清晰准确的回答。",
    "Python 程序需要经过测试才能可靠运行。",
    "请总结这段文字并列出三个关键观点。",
    "机器学习模型从训练数据中寻找统计规律。",
    "这个系统包含前端、后端、数据库和任务队列。",
    "遇到错误时，先阅读日志，再定位输入和环境。",
    "大语言模型根据上文预测下一个 Token。",
    "我们需要比较实验结果，而不是只凭直觉判断。",
    "今天完成实验以后，把结论写进知识库。",
]

MATERIAL_SENTENCES = [
    "MagNova 根据用户需求生成并筛选候选材料。",
    "MatterGen 负责生成新的无机晶体结构。",
    "MatterSim 对候选结构执行快速预弛豫。",
    "SACGT 预测材料的稳定性、磁矩和磁矩密度。",
    "请设计高磁性、低重稀土依赖的 Fe-Co-B 材料。",
    "Nd2Fe14B 是一种典型的永磁材料化学式。",
    "候选材料需要经过 OOD 检测和不确定性分析。",
    "目标磁矩密度设为 0.2 μB/Å³，最大原子数为 20。",
    "Agent 将自然语言需求转换成结构化工具调用。",
    '<tool_call>{"name":"submit_magnova_task"}</tool_call>',
]

TEST_SENTENCES = [
    "今天天气很好，我们学习人工智能。",
    "请设计高磁性低重稀土依赖的 Fe-Co-B 材料。",
    "MatterGen 生成磁矩密度为 0.2 μB/Å³ 的候选结构。",
    '<tool_call>{"name":"submit_magnova_task"}</tool_call>',
]

SPECIAL_TOKENS = [
    "<unk>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
]


def make_corpus(sentences, repeats):
    result = []
    for _ in range(repeats):
        result.extend(sentences)
    return result


def make_mixed_corpus(repeats):
    result = []
    for _ in range(repeats):
        result.extend(GENERAL_SENTENCES)
        result.extend(MATERIAL_SENTENCES)
    return result


def train_tokenizer(corpus, vocab_size=400):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=False,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer


def describe(name, tokenizer, text):
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
    print(f"  {name}")
    print(f"    token_count: {len(encoded.ids)}")
    print(f"    byte_per_token: {len(text.encode('utf-8')) / len(encoded.ids):.2f}")
    print(f"    tokens: {' | '.join(encoded.tokens)}")
    print(f"    roundtrip_equal: {decoded == text}")
    return len(encoded.ids)


def main():
    # Both corpora contain 300 lines. Only the data distribution differs.
    general_corpus = make_corpus(GENERAL_SENTENCES, repeats=30)
    mixed_corpus = make_mixed_corpus(repeats=15)

    general_tokenizer = train_tokenizer(general_corpus)
    mixed_tokenizer = train_tokenizer(mixed_corpus)

    print("=== 实验控制 ===")
    print(f"通用语料行数: {len(general_corpus)}")
    print(f"混合语料行数: {len(mixed_corpus)}")
    print(f"目标词表大小: 400")
    print(f"通用 Tokenizer 实际词表: {general_tokenizer.get_vocab_size()}")
    print(f"材料增强 Tokenizer 实际词表: {mixed_tokenizer.get_vocab_size()}")

    print("\n=== 同一文本的对照结果 ===")
    for text in TEST_SENTENCES:
        print(f"\n文本: {text}")
        general_count = describe("通用语料 Tokenizer", general_tokenizer, text)
        mixed_count = describe("材料增强 Tokenizer", mixed_tokenizer, text)
        delta = general_count - mixed_count
        print(f"  材料增强后少用 Token: {delta}")

    print("\n=== 结论边界 ===")
    print("这个实验只测量小语料上的压缩变化，不代表下游模型能力。")
    print("真实决策还要比较通用测试集、材料测试集和 Agent 任务指标。")


if __name__ == "__main__":
    main()
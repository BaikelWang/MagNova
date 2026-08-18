# MagNova project status

Last verified: 2026-08-18

This document is a public-safe progress snapshot. It records engineering outcomes without publishing model weights, private corpora, sealed evaluation sets, candidate structure files, or SACGT algorithm details.

## Current system baseline

| Component | Verified state | Public boundary |
|---|---|---|
| MiniMind requirement parser | Deployed through an OpenAI-compatible endpoint; schema/slot guardrails remain active | Runtime and adapters are public; weights and full datasets are private |
| Evidence-grounded research assistant | Frozen research-answer baseline is deployed with BGE retrieval, paper evidence, citations and refusal gates | RAG runtime and evaluation code are public; corpus and weights are private |
| Agent workflow | LangGraph routes research discussion, requirement clarification, human approval and tool execution | Public |
| Scientific backend | Mock, HTTP and private CLI modes share a typed contract; real failures never fall back to mock | Adapter contract is public; SACGT core is private |
| Web console | Conversation, candidate, provenance, metrics and system pages; dark and presentation modes | Public |
| Closed-loop artifacts | Funnel, ranking, raw/relaxed structure views and proxy charts generated from completed runs | Selected figures only |
| Quality | 299 tests passed, 1 skipped; end-to-end stack smoke passed | Public tests are a subset of the internal verification estate |

## What the team added in August 2026

### 1. Evidence-grounded scientific dialogue

The research path now separates three behaviors:

- `positive_answer`: the evidence supports an affirmative answer;
- `grounded_negative`: the evidence supports a negative conclusion;
- `insufficient_evidence`: the available corpus cannot support an answer.

This avoids treating a supported negative conclusion as a refusal and makes the boundary between “known false” and “not evidenced” explicit.

The runtime now includes:

- BGE-compatible dense retrieval;
- sentence-level evidence packing;
- multi-turn query condensation;
- source/citation extraction;
- answerability and faithfulness gates;
- claim-to-evidence validation;
- out-of-domain refusal;
- auditable failure analysis.

### 2. Evidence-verifier research

The team investigated why evidence sufficiency was difficult to classify:

- whole-document cross-encoding truncated most examples at the 512-token backbone limit;
- teacher review showed that part of the automatically generated counterfactual data had noisy labels;
- chunk-level multiple-instance learning substantially reduced truncation.

The best train-only candidate used mean pooling over independently encoded evidence chunks:

| Experiment | Question-grouped OOF balanced accuracy | Decision |
|---|---:|---|
| Whole evidence, ordinary training | 61.90% | Rejected |
| Whole evidence, staged unfreezing | 75.08% | Rejected |
| Teacher-corrected labels | 70.56% | Rejected |
| Chunk encoding + max pooling | 77.64% | Rejected |
| Chunk encoding + mean pooling | **78.38%** | Rejected: below 80% gate |

No candidate replaced the frozen research-answer baseline. Blind evaluation remained sealed for these experiments.

### 3. Agent and backend integration

The Agent now supports:

- stable `MaterialRequest`, `JobSnapshot` and `CandidateMaterial` contracts;
- a human confirmation interrupt before scientific tool calls;
- a deterministic mock backend for product and contract testing;
- an asynchronous HTTP backend;
- a CLI bridge to the private MagNova/SACGT workflow;
- explicit blocked and needs-clarification states;
- no silent fallback from real execution to mock output;
- redacted telemetry and task-state persistence.

The scientific team also added a JSON request entry point and reproducible closed-loop launch scripts around the private backend. Core SACGT implementation and proprietary assets remain outside this repository.

### 4. Visualization and product experience

The Streamlit console now includes:

- conversation history and multiple sessions;
- research evidence and retrieval provenance;
- a visible human-approval gate;
- candidate cards and deterministic result tables;
- screening funnel and confidence matrix;
- raw/relaxed crystal views;
- read-only artifact loading;
- light, dark and presentation modes;
- explicit Mock/Real/AI-screening labels.

Selected public figures are stored in `docs/assets/showcase/`.

## Verification evidence

The authoritative development workspace was verified with:

```text
299 passed, 1 skipped
```

The live-stack smoke covered:

1. a scientific question with cited paper evidence;
2. a multi-turn follow-up with resolved context;
3. refusal when evidence is insufficient;
4. natural-language design intent converted to `MaterialRequest`;
5. human approval;
6. handoff to the mock scientific pipeline.

All checks passed. Browser-based visual regression remains a separate manual release check.

## Scientific interpretation

A generated structure is not automatically:

- thermodynamically or dynamically stable;
- ferromagnetic in its ground state;
- synthesizable;
- novel;
- suitable for an industrial application.

MatterSim-style pre-relaxation, SACGT proxy prediction, uncertainty ranking, DFT and experiment represent different evidence levels. MagNova keeps those levels distinct in code, UI and documentation.

## Next gates

- reach the evidence-verifier acceptance threshold without consuming sealed evaluation data;
- convert the private CLI bridge into a production asynchronous HTTP service;
- calibrate uncertainty and OOD thresholds on held-out scientific data;
- implement Pareto and diversity-aware selection;
- attach DFT provenance and failure reasons to each candidate dossier;
- add experimental feedback and active-learning loops.

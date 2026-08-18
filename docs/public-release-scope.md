# Public release scope

This repository is a curated project showcase, not a full research artifact dump.

## Included

- Agent runtime, schemas, conversation logic, orchestration, persistence, and UI;
- abstract generator/predictor/screener interfaces and mock implementations;
- HTTP and CLI adapter contracts that do not expose private backends;
- evidence-grounded RAG runtime, validation, evaluation, and failure-analysis code;
- generic MiniMind SFT, evaluation, tokenizer, embedding, and export utilities;
- MatterGen/MatterSim workflow scripts that rely on separately obtained upstream assets;
- small golden examples, aggregate metrics, selected non-sensitive showcase figures, tests, and engineering notes.

## Excluded

- all MiniMind, MatterGen, MatterSim, SACGT, ALIGNN, and other model weights;
- SACGT architecture, training, inference, attribution, and uncertainty implementation;
- full SFT corpora, hidden/adversarial test sets, LMDB databases, CIF collections, and raw predictions;
- API keys, SSH credentials, private endpoints, server addresses, local absolute paths, and task databases;
- logs, PID files, caches, checkpoints, archives, and generated trajectories;
- unpublished experimental results whose release could disclose core scientific innovation.

## Integration principle

Private scientific components connect only through typed adapter contracts. A public caller can understand the system boundary and reproduce the mock workflow without receiving private algorithms or assets.

Before every release, run both a secret scan and a large-file scan. The root `.gitignore` is defense in depth and is not a replacement for reviewing the staged Git tree.

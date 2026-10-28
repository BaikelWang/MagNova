# Public release scope

This repository is a curated project showcase, not a full research artifact dump.

## Competition showcase — October 2026

MagNova received a National First Prize and reached the national Top 20 in the 2026 Huawei Cup China Postgraduate AI Innovation Competition. The October 28 showcase update adds the project story, original architecture graphics, selected historical UI screenshots, and bounded validation summaries. It does **not** replace the existing public application code or publish the entire competition deployment.

The distinction is deliberate:

- **Competition system:** the deployed research workflow shown in the screenshots, including authenticated access, evidence retrieval, explicit confirmation, real scientific execution, candidate provenance and recovery.
- **Public code baseline:** the runnable contracts, agent components, task-management interfaces, mock workflow, training utilities and selected integration experiments already present here.
- **Showcase evidence:** historical same-task extracts and aggregate acceptance records, with dates, limitations and media attribution. This is not a fresh run, a new model evaluation, or an assertion that the hosted service remains online.

No award certificate, personal contact information, account credential or private manuscript is included. Screenshots show task/candidate identifiers for traceability; these are not login or execution credentials. See [the case](showcase-case.md), [validation boundaries](validation.md) and [media attribution](assets/README.md).

## Included

- Agent runtime, schemas, conversation logic, orchestration, persistence, and UI;
- abstract generator/predictor/screener interfaces and mock implementations;
- HTTP integration contracts that do not expose private backends;
- generic MiniMind SFT, evaluation, tokenizer, embedding, and export utilities;
- MatterGen/MatterSim workflow scripts that rely on separately obtained upstream assets;
- small golden examples, aggregate parsing metrics, tests, and engineering notes.

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

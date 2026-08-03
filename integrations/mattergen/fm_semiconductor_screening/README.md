# MatterGen magnetic-material generation and stability screening

This public workflow stops after MatterSim stability screening. The incomplete
ALIGNN experiment and its downloaded assets are intentionally not published.

The output structures are **high-magnetic-density conditioned,
MatterSim-screened candidates**. They are not confirmed ferromagnets or
semiconductors; those claims require electronic-structure and magnetic-order
calculations beyond this workflow.

## 1. Install MatterGen

Copy this directory into a full MatterGen checkout as
`workflows/fm_semiconductor_test32`, then install MatterGen:

```bash
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

The local LMDB compatibility fix in
`mattergen/evaluation/reference/reference_dataset_serializer.py` must also be
present when using a python-lmdb version that rejects reopening one path in a
single process.

## 2. Prepare local assets

Download the magnetic-density checkpoint and TRI2024 reference once:

```bash
bash workflows/fm_semiconductor_test32/00_download_assets.sh
```

The generation stage uses only these local files:

- `checkpoints/dft_mag_density/checkpoints/last.ckpt`
- `checkpoints/dft_mag_density/config.yaml`
- `data-release/alex-mp/reference_TRI2024correction.gz`

The download script validates their exact sizes and SHA-256 checksums.

If the GPU server is offline, prepare them on another machine and copy this
archive into the MatterGen repository root:

```bash
tar -czf mattergen-local-assets.tar.gz \
  checkpoints/dft_mag_density \
  data-release/alex-mp/reference_TRI2024correction.gz

tar -xzf mattergen-local-assets.tar.gz
```

## 3. Generate and screen 32 structures

Activate the MatterGen environment and run:

```bash
source .venv/bin/activate
bash workflows/fm_semiconductor_test32/01_generate_evaluate.sh \
  2>&1 | tee fm_test32.log
```

Defaults:

- 32 structures (`batch_size=16`,
um_batches=2`).
- Target magnetic density: `0.20 A^-3`.
- Diffusion guidance factor: `2.0`.
- MatterSim-v1-1M relaxation.
- TRI2024 stability reference.
- Stability threshold: `energy_above_hull <= 0.10 eV/atom`.

For a 16-structure test:

```bash
BATCH_SIZE=16 NUM_BATCHES=1 \
  bash workflows/fm_semiconductor_test32/01_generate_evaluate.sh
```

If generation has completed and evaluation needs to be resumed:

```bash
SKIP_GENERATION=1 \
  bash workflows/fm_semiconductor_test32/01_generate_evaluate.sh
```

## Outputs

All outputs are under `results/fm_semiconductor_test32`:

- `generated_crystals.extxyz`: generated structures.
- `relaxed.extxyz`: MatterSim-relaxed structures.
- `metrics.json`: aggregate evaluation metrics.
- `detailed_metrics.json`: per-structure metrics.
- `mattergen_screening.csv`: all submitted structures and pass/fail reasons.
- `stable_candidates.csv`: stable, unique, structurally valid candidates,
  ordered by novelty and energy above the hull.
- `stable_cifs/`: one CIF for every row in `stable_candidates.csv`.

Inspect the final candidates with:

```bash
head -n 20 results/fm_semiconductor_test32/stable_candidates.csv
ls results/fm_semiconductor_test32/stable_cifs
```

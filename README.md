# StructCodec

Implementation of **Critical-structure rate–distortion through persistent contour transport.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run. GUDHI runtime dependencies were not installed in this inspection environment.

## Data and preprocessing

Download the public [BSDS500 release](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) using the included script. It preserves official train/validation/test image identities and writes portable paths plus `source.json`. The codec uses only train rows for fitting. Cite Arbelaez et al., Contour Detection and Hierarchical Image Segmentation, TPAMI 2011, as requested by the data provider.

```bash
python prepare_bsds500.py --output data/bsds500
python -m structcodec prepare --input data/bsds500/images.csv --config config.json --output runs/full
```

For an existing archive use `python prepare_bsds500.py --offline --archive data/BSR_bsds500.tgz --output data/bsds500`. Custom image manifests use `image,split` and paths relative to the CSV; optional annotation/mask/depth fields are consumed only by downstream task evaluators.

## Training and inference

```bash
python -m structcodec train --config config.json --output runs/full
python -m structcodec run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m structcodec evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `uniform`, `permuted`, `edge-only`.

Repeat the same data and configuration in a separate directory, for example:

```bash
python -m structcodec prepare --input data/images.csv --config config.json --output runs/uniform --variant uniform
python -m structcodec train --config config.json --output runs/uniform --variant uniform
python -m structcodec run --config config.json --output runs/uniform --variant uniform
python -m structcodec evaluate --config config.json --output runs/uniform --variant uniform
```

## Outputs

All main outputs are written below `runs/full/`.

`decoded/*.scz` are independently decodable compressed streams; `decoded/*.png` are reconstructions. Stream size includes integer latent symbols, the uint8 spatial quantization map and container headers. `predictions.csv` contains actual bpp, PSNR, eligible/broken structures and coding time. `metrics.csv` aggregates fidelity/structure results. Frozen-task and BD-rate commands write separate JSON metrics.

```bash
python -m structcodec plot --config config.json --output runs/full
```

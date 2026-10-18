# StructCodec

Implementation of **Critical-structure rate–distortion through persistent contour transport.**

## Install

Use Python 3.11 in an isolated environment. Run from the repository root:

```bash
python -m pip install -e .
```

## Data

Prepare an image CSV with `image,split` (`train,test`) and paths relative to the manifest. Optional `annotation`, `mask` and `depth` fields support frozen downstream evaluation. The paper uses Kodak/Tecnick/CLIC for fidelity and COCO/ADE20K/NYU Depth v2 for tasks. Detection annotation JSON uses `boxes` in xyxy pixels and torchvision COCO label IDs. Segmentation masks use zero-based ADE classes 0–149 and ignore 255. Depth NPY arrays are metric depth in meters.

## Workflow

```bash
python -m structcodec prepare --input data/images.csv
python -m structcodec train
python -m structcodec run
python -m structcodec evaluate
python -m structcodec plot
```

Main commands accept `--config config.json` and `--output runs/default`. Each output directory records `settings.json`. Keep independent seeds, datasets and conditions in separate directories.

## Configuration

`base_step` sets quantization; `allocation_gain` controls criticality modulation. `pixel_weight,structure_weight,orientation_weight` define distortion. `pyramid,events` bound multiscale cubical persistence events. `sinkhorn_epsilon,sinkhorn_iterations,displacement_weight` define event transport. `persistence_cutoff,matching_tolerance` define eligible structures and survival matching. Input size must be divisible by 16. Additional optimizer, batch-size and model-width defaults are in `config.json`.

## Outputs

`decoded/*.scz` are independently decodable compressed streams; `decoded/*.png` are reconstructions. Stream size includes integer latent symbols, the uint8 spatial quantization map and container headers. `predictions.csv` contains actual bpp, PSNR, eligible/broken structures and coding time. `metrics.csv` aggregates fidelity/structure results. Frozen-task and BD-rate commands write separate JSON metrics.

## Method and comparisons

GUDHI cubical persistence selects critical cells; gradients pass through the selected birth/death values. Entropic transport includes insertion/deletion costs and spatial support displacement. The detached source distortion gradient sets a transmitted quantization map. Analysis/synthesis transforms are compact four-stage convolutional networks, with a Gaussian likelihood training rate and lossless Deflate integer-stream coding. Actual bpp is measured from the complete stream, rather than inferred from the likelihood. Structural counts use the configured eligible event set.

Controls: `full`, `uniform`, `permuted`, `edge-only`. Choose `--variant` and use a separate training/output directory.

```bash
python -m structcodec.compare --inputs "runs/*/metrics.csv" --reference full --metrics bpp psnr structural_break_rate --unit seed
```

Comparison averages rows within independent units before bootstrap intervals and paired Wilcoxon/Holm inference. For participant/client analyses choose those independent unit columns instead of counting recordings as replicates.

## Frozen task evaluation and rate curves

```bash
python -m pip install -e ".[evaluation]"
python -m structcodec.downstream detection --input runs/default/predictions.csv --output runs/detection.json
python -m structcodec.downstream segmentation --input runs/default/predictions.csv --output runs/segmentation.json
python -m structcodec.downstream depth --input runs/default/predictions.csv --output runs/depth.json
python -m structcodec.downstream bd-rate --input runs/rate_curve.csv --reference runs/baseline_curve.csv --output runs/bd_rate.json
```

Detection uses frozen Faster R-CNN/COCO mAP, segmentation uses a frozen ADE20K SegFormer with mIoU, and depth uses a frozen NYU GLPN with metric RMSE. Rate-curve CSVs contain at least four distinct `bpp,psnr` points; BD-rate integrates log rate over common PSNR support. Sweep `base_step` using frozen codec weights and record complete stream bpp to select operating points. Preserve the same eligible image/structure panel across methods.

## Reproduction logic conventions

Structure-survival matching first maximizes the number of pairs satisfying persistence and matching-tolerance requirements, then minimizes distance among those pairs. Ineligible matches receive a sufficiently large assignment cost; a cheap invalid pair cannot consume a valid alternative.

## Selected executable codec setting

`config.json` fixes RGB 256-by-256 images, width 64, the four-stage learned
analysis/synthesis transform, base_step=0.1, allocation_gain=1 and uint8 transmitted
criticality map. Gaussian likelihood trains the rate proxy; the actual stream is
int16 symbols plus the map in a losslessly Deflate-compressed NPZ container.
Complete container bytes define bpp. This is the chosen codec/entropy setting;
a different neural entropy backend or external compression model is unnecessary.

```bash
python prepare_bsds500.py --plan
python prepare_bsds500.py --output data/bsds500
python -m structcodec prepare --input data/bsds500/images.csv --config config.json --output runs/deflate
python -m structcodec train --config config.json --output runs/deflate --variant full
python -m structcodec run --config config.json --output runs/deflate
python -m structcodec evaluate --config config.json --output runs/deflate
```

The manifest supplies disjoint train/test image paths. Train and decode the same
weights; the side map and quantization parameters are carried in each stream.
Uniform/permuted/edge-only controls retain this exact coding container. Fidelity,
actual bpp and thresholded structural survival are the primary endpoints; frozen
detection/segmentation/depth models and their datasets are optional downstream
endpoints and are not downloaded by the codec workflow.

`prepare_bsds500.py` acquires the public [BSDS500 release](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
directly from Berkeley. It reads only regular image members and preserves the
official 200 train, 100 validation and 200 test identities in `images.csv`.
Paths are relative to the manifest, so the prepared data directory is portable.
Only train rows fit the codec; validation is reserved for choosing configurations
and test is evaluated after settings are frozen. No segmentation annotations or
external pretrained codec are needed: structural fidelity is computed between
each original image and its decoded reconstruction. The existing codec loader
converts to RGB and applies the configured 256px square resize consistently.
`source.json` records acquisition and preprocessing. A local official archive
can be used with `--offline --archive data/BSR_bsds500.tgz`.

Please cite Arbelaez et al., *Contour Detection and Hierarchical Image
Segmentation*, TPAMI 2011, as requested on the dataset source page. The download
script retains the original data; the repository's code license does not
relicense those images.

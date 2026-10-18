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

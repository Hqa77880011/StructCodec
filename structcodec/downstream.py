import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument('task', choices=['detection', 'segmentation', 'depth', 'bd-rate'])
    p.add_argument('--input', required=True)
    p.add_argument('--reference')
    p.add_argument('--output', default='runs/downstream.json')
    args = p.parse_args()
    frame = pd.read_csv(args.input)
    if args.task == 'bd-rate':
        from scipy.interpolate import PchipInterpolator
        baseline = pd.read_csv(args.reference).sort_values('psnr')
        curve = frame.sort_values('psnr')
        lower = max(baseline.psnr.min(), curve.psnr.min())
        upper = min(baseline.psnr.max(), curve.psnr.max())
        if upper <= lower:
            raise ValueError('BD-rate needs overlapping PSNR support')
        a = PchipInterpolator(baseline.psnr, np.log(baseline.bpp)).integrate(lower, upper)
        b = PchipInterpolator(curve.psnr, np.log(curve.bpp)).integrate(lower, upper)
        result = dict(bd_rate_percent=100 * (np.exp((b - a) / (upper - lower)) - 1))
    elif args.task == 'detection':
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
        from torchvision.transforms.functional import to_tensor
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
        metric = MeanAveragePrecision()
        for row in frame.itertuples():
            target = json.loads(Path(row.annotation).read_text(encoding='utf-8'))
            original = Image.open(row.image).convert('RGB')
            decoded = Image.open(row.decoded).convert('RGB').resize(original.size)
            with torch.no_grad():
                prediction = model([to_tensor(decoded)])
            metric.update(prediction, [dict(boxes=torch.tensor(target['boxes'], dtype=torch.float32), labels=torch.tensor(target['labels'], dtype=torch.long))])
        result = {k: float(v) for k, v in metric.compute().items() if v.ndim == 0}
    elif args.task == 'segmentation':
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        checkpoint = 'nvidia/segformer-b0-finetuned-ade-512-512'
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint).eval()
        confusion = np.zeros((150, 150), dtype=np.int64)
        for row in frame.itertuples():
            truth = np.asarray(Image.open(row.mask), dtype=np.int64)
            inputs = processor(images=Image.open(row.decoded).convert('RGB'), return_tensors='pt')
            with torch.no_grad():
                logits = model(**inputs).logits
            prediction = F.interpolate(logits, truth.shape, mode='bilinear', align_corners=False).argmax(1)[0].numpy()
            valid = (truth >= 0) & (truth < 150)
            confusion += np.bincount(150 * truth[valid] + prediction[valid], minlength=150 ** 2).reshape(150, 150)
        denominator = confusion.sum(0) + confusion.sum(1) - confusion.diagonal()
        result = dict(miou=float(np.mean(confusion.diagonal()[denominator > 0] / denominator[denominator > 0])))
    else:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        checkpoint = 'vinvino02/glpn-nyu'
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = AutoModelForDepthEstimation.from_pretrained(checkpoint).eval()
        squared, count = 0., 0
        for row in frame.itertuples():
            truth = np.load(row.depth)
            inputs = processor(images=Image.open(row.decoded).convert('RGB'), return_tensors='pt')
            with torch.no_grad():
                predicted = model(**inputs).predicted_depth[:, None]
            predicted = F.interpolate(predicted, truth.shape, mode='bilinear', align_corners=False)[0, 0].numpy()
            valid = np.isfinite(truth) & (truth > 0)
            squared += float(((predicted[valid] - truth[valid]) ** 2).sum())
            count += int(valid.sum())
        result = dict(depth_rmse=np.sqrt(squared / count))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))


if __name__ == '__main__':
    main()

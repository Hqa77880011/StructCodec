import argparse
import io
import json
import time
from pathlib import Path

import gudhi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


def diagram(field, limit):
    array = field.detach().cpu().numpy()
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=array)
    complex_.persistence()
    regular, _ = complex_.cofaces_of_persistence_pairs()
    pairs = np.concatenate([p for p in regular if len(p)], axis=0) if any(len(p) for p in regular) else np.empty((0, 2), dtype=int)
    if not len(pairs):
        return field.new_empty((0, 4))
    flat = field.T.reshape(-1)
    pair_ids = torch.tensor(pairs, dtype=torch.long, device=field.device)
    birth, death = flat[pair_ids[:, 0]], flat[pair_ids[:, 1]]
    selected = (death - birth).abs().topk(min(limit, len(pairs))).indices
    height, width = field.shape
    coords = torch.stack([(pair_ids[:, 0] % height).float() / height, (pair_ids[:, 0] // height).float() / width], 1)
    return torch.cat([birth[:, None], death[:, None], coords], 1)[selected]


def transport(a, b, c):
    if not len(a):
        return .5 * (b[:, 1] - b[:, 0]).abs().sum()
    if not len(b):
        return .5 * (a[:, 1] - a[:, 0]).abs().sum()
    cost = torch.cdist(a[:, :2], b[:, :2], p=1) + c['displacement_weight'] * torch.cdist(a[:, 2:], b[:, 2:])
    deletion = .5 * (a[:, 1] - a[:, 0]).abs()
    insertion = .5 * (b[:, 1] - b[:, 0]).abs()
    augmented = torch.cat([torch.cat([cost, deletion[:, None].expand(-1, len(a))], 1),
                           torch.cat([insertion[None].expand(len(b), -1), cost.new_zeros(len(b), len(a))], 1)], 0)
    logk = -augmented / c['sinkhorn_epsilon']
    n = len(augmented)
    u = torch.zeros(n, device=a.device)
    v = u.clone()
    logmass = -np.log(n)
    for _ in range(c['sinkhorn_iterations']):
        u = logmass - torch.logsumexp(logk + v[None], 1)
        v = logmass - torch.logsumexp(logk + u[:, None], 0)
    plan = torch.exp(logk + u[:, None] + v[None])
    return n * (plan * augmented).sum()


def gradient(x):
    horizontal = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1))
    vertical = F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 0, 0, 1))
    return horizontal, vertical


def structural(source, reconstruction, c):
    loss = source.sum() * 0
    for scale in c['pyramid']:
        a = F.avg_pool2d(source.mean(1, keepdim=True), scale)
        b = F.avg_pool2d(reconstruction.mean(1, keepdim=True), scale)
        for image_a, image_b in zip(a[:, 0], b[:, 0]):
            for sign in [-1, 1]:
                loss = loss + transport(diagram(sign * image_a, c['events']), diagram(sign * image_b, c['events']), c)
    gx, gy = gradient(source)
    hx, hy = gradient(reconstruction)
    orientation = 1 - (gx * hx + gy * hy) / ((gx.square() + gy.square() + 1e-8).sqrt() * (hx.square() + hy.square() + 1e-8).sqrt())
    return loss / len(source) + c['orientation_weight'] * orientation.mean()


class Codec(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        channels = [3, c['width'], c['width'], c['width'], c['width']]
        analysis, synthesis = [], []
        for i in range(4):
            analysis += [nn.Conv2d(channels[i], channels[i + 1], 5, stride=2, padding=2)]
            if i != 3:
                analysis += [nn.ReLU()]
        for i in range(4):
            synthesis += [nn.ConvTranspose2d(channels[4 - i], channels[3 - i], 5, stride=2, padding=2, output_padding=1)]
            synthesis += [nn.Sigmoid() if i == 3 else nn.ReLU()]
        self.analysis = nn.Sequential(*analysis)
        self.synthesis = nn.Sequential(*synthesis)
        self.log_scale = nn.Parameter(torch.zeros(1, c['width'], 1, 1))

    def quantization(self, x):
        with torch.enable_grad():
            source = x.detach().clone().requires_grad_(True)
            latent = self.analysis(source)
            coarse = self.synthesis(torch.round(latent / self.c['base_step']) * self.c['base_step']).detach()
            if self.c['variant'] == 'edge-only':
                gx, gy = gradient(source)
                hx, hy = gradient(coarse)
                distortion = (gx - hx).abs().mean() + (gy - hy).abs().mean()
            else:
                distortion = structural(source, coarse, self.c)
            criticality = torch.autograd.grad(distortion, source)[0].abs().mean(1, keepdim=True)
        criticality = F.adaptive_avg_pool2d(criticality, latent.shape[-2:]).detach()
        criticality /= criticality.amax((-2, -1), keepdim=True).clamp_min(1e-8)
        if self.c['variant'] == 'uniform':
            criticality = criticality * 0
        elif self.c['variant'] == 'permuted':
            order = torch.randperm(criticality.shape[-2] * criticality.shape[-1], device=x.device)
            criticality = criticality.flatten(2)[:, :, order].reshape_as(criticality)
        indices = (criticality * 255).round().to(torch.uint8)
        step = self.c['base_step'] * torch.exp(-self.c['allocation_gain'] * indices.float() / 255)
        return step, indices

    def forward(self, x):
        step, side = self.quantization(x)
        latent = self.analysis(x)
        symbols = latent / step
        quantized = symbols + (symbols.round() - symbols).detach()
        reconstruction = self.synthesis(quantized * step)
        scale = F.softplus(self.log_scale) + 1e-5
        upper = torch.special.ndtr((quantized + .5) * step / scale)
        lower = torch.special.ndtr((quantized - .5) * step / scale)
        rate = -torch.log2((upper - lower).clamp_min(1e-9)).sum() / (len(x) * x.shape[-2] * x.shape[-1])
        rate = rate + 8 * side.numel() / (len(x) * x.shape[-2] * x.shape[-1])
        return reconstruction, rate


def image(path, c):
    raw = np.asarray(Image.open(path).convert('RGB').resize((c['size'], c['size'])), dtype=np.float32) / 255
    return torch.tensor(raw).permute(2, 0, 1).unsqueeze(0)


def prepare(c, out, source):
    source = Path(source)
    frame = pd.read_csv(source)
    if not {'image', 'split'} <= set(frame):
        raise ValueError('Manifest requires image,split with optional mask,depth,annotation columns')
    for key in ['image', 'mask', 'depth', 'annotation']:
        if key in frame:
            frame[key] = frame[key].fillna('').map(lambda p: str((source.parent / p).resolve()) if p else '')
    if frame.image.duplicated().any():
        raise ValueError('Image identities must be disjoint')
    frame.to_csv(out / 'manifest.csv', index=False)


def train(c, out):
    torch.manual_seed(c['seed'])
    model = Codec(c)
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
    frame = pd.read_csv(out / 'manifest.csv')
    training = frame[frame.split == 'train']
    curve = []
    for epoch in range(c['epochs']):
        losses = []
        for path in training.sample(frac=1, random_state=c['seed'] + epoch).image:
            source = image(path, c)
            reconstruction, rate = model(source)
            distortion = structural(source, reconstruction, c)
            if c['variant'] == 'edge-only':
                gx, gy = gradient(source)
                hx, hy = gradient(reconstruction)
                distortion = (gx - hx).abs().mean() + (gy - hy).abs().mean()
            loss = rate + c['pixel_weight'] * F.mse_loss(reconstruction, source) + c['structure_weight'] * distortion
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        curve.append(dict(epoch=epoch, objective=np.mean(losses)))
    torch.save(model.state_dict(), out / 'model.pt')
    pd.DataFrame(curve).to_csv(out / 'learning.csv', index=False)


def encode(model, source):
    step, side = model.quantization(source)
    with torch.no_grad():
        rounded = torch.round(model.analysis(source) / step)
    if rounded.abs().max() > 32767:
        raise OverflowError('Quantized symbol exceeds int16 coding range')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, symbols=rounded.numpy().astype(np.int16), side=side.numpy(),
                        step=np.array(model.c['base_step']), gain=np.array(model.c['allocation_gain']))
    return buffer.getvalue()


def decode(model, stream):
    with np.load(io.BytesIO(stream), allow_pickle=False) as coded:
        step = float(coded['step']) * np.exp(-float(coded['gain']) * coded['side'].astype(float) / 255)
        latent = torch.tensor(coded['symbols'] * step, dtype=torch.float32)
    with torch.no_grad():
        return model.synthesis(latent)


def run(c, out):
    model = Codec(c)
    model.load_state_dict(torch.load(out / 'model.pt', weights_only=True, map_location='cpu'))
    model.eval()
    frame = pd.read_csv(out / 'manifest.csv').fillna('')
    destination = out / 'decoded'
    destination.mkdir(exist_ok=True)
    rows = []
    for i, row in enumerate(frame[frame.split == 'test'].itertuples()):
        source = image(row.image, c)
        start = time.perf_counter()
        stream = encode(model, source)
        reconstruction = decode(model, stream)
        elapsed = time.perf_counter() - start
        (destination / f'{i}.scz').write_bytes(stream)
        decoded = destination / f'{i}.png'
        Image.fromarray((reconstruction[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)).save(decoded)
        mse = float((source - reconstruction).square().mean())
        source_events = diagram(F.avg_pool2d(source.mean(1), c['pyramid'][0])[0], c['events']).detach().numpy()
        decoded_events = diagram(F.avg_pool2d(reconstruction.mean(1), c['pyramid'][0])[0], c['events']).detach().numpy()
        eligible = source_events[np.abs(source_events[:, 1] - source_events[:, 0]) >= c['persistence_cutoff']]
        survived = 0
        if len(eligible) and len(decoded_events):
            cost = np.linalg.norm(eligible[:, None] - decoded_events[None], axis=2)
            a, b = linear_sum_assignment(cost)
            survived = int(((np.abs(decoded_events[b, 1] - decoded_events[b, 0]) >= c['persistence_cutoff']) &
                            (cost[a, b] <= c['matching_tolerance'])).sum())
        result = row._asdict()
        result.update(decoded=str(decoded.resolve()), bpp=8 * len(stream) / c['size'] ** 2,
                      psnr=-10 * np.log10(max(mse, 1e-12)), eligible_structures=len(eligible),
                      structural_breaks=len(eligible) - survived, coding_seconds=elapsed)
        rows.append(result)
    pd.DataFrame(rows).to_csv(out / 'predictions.csv', index=False)


def evaluate(c, out):
    d = pd.read_csv(out / 'predictions.csv')
    result = dict(method=c['variant'], bpp=d.bpp.mean(), psnr=d.psnr.mean(),
                  structural_breaks=int(d.structural_breaks.sum()), eligible_structures=int(d.eligible_structures.sum()),
                  structural_break_rate=d.structural_breaks.sum() / max(1, d.eligible_structures.sum()),
                  coding_seconds=d.coding_seconds.mean())
    pd.DataFrame([result]).to_csv(out / 'metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'predictions.csv')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d.bpp, d.psnr)
    ax.set(xlabel='Complete bitstream bpp', ylabel='PSNR (dB)')
    fig.tight_layout()
    fig.savefig(out / 'rate_distortion.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'train', 'run', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    p.add_argument('--variant')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.variant:
        c['variant'] = args.variant
    if c['variant'] not in ['full', 'uniform', 'permuted', 'edge-only']:
        p.error('Unknown variant')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    if args.variant is None and settings.exists():
        c['variant'] = json.loads(settings.read_text(encoding='utf-8')).get('variant', c.get('variant', 'full'))
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if not args.input:
            p.error('--input required')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()

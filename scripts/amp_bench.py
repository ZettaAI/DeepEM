#!/usr/bin/env python
"""Mixed-precision parity and speed/memory benchmark on one GPU.

Builds the exact training options from a Seuron training spec (the same JSON
that is uploaded to Slack), then, on synthetic data of the spec's patch size
and heads, runs the real model wrapper, losses and optimizer:

  --parity      one step from identical weights and input: per-head loss
                relative error and gradient cosine of bf16/fp16 against fp32.
                The cosine is judged against a floor measured in fp32 itself
                by perturbing the input at the 16-bit dtype's relative
                precision -- at random init the gradient direction is that
                sensitive, so 0.9 can be fine and a fixed 0.99 bar is not.
  --trajectory  N training steps from identical weights on a fixed cycle of
                synthetic samples, in fp32 twice (the cuDNN nondeterminism
                floor) and in each 16-bit mode; compares the loss curves.
  --bench       s/iter (median after warmup) and peak allocated memory for
                every combination of --widths x --modes x --batch x
                --channels_last. OOM is recorded, not fatal.

Data loading is not included, so s/iter here is a lower bound on training.

    python scripts/amp_bench.py --spec riser/train/riser_v2_long.json \\
        --parity --trajectory 200 --bench --out amp_bench_out

Writes <out>/results.json and <out>/results.md.
"""

import argparse
import copy
import json
import math
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Seuron-only keys, and keys the launcher itself sets (as in riser's smoke_train.sh).
SEURON_KEYS = ["remote_dir", "deepem_image", "TORCHRUN_LAUNCHER", "NUM_TRAINERS",
               "samwise_period", "WANDB_API_KEY", "skip_export", "gpu_ids"]
EPS = {'bf16': 2.0 ** -8, 'fp16': 2.0 ** -11}


def spec_opt(spec_path):
    """Parse the training spec with DeepEM's own Options, as run.py would."""
    from deepem.train.option import Options

    spec = json.load(open(spec_path))
    for k in SEURON_KEYS:
        spec.pop(k, None)

    def fmt(k, v):
        if v is None:
            return [f"--{k}"]
        if isinstance(v, list):
            return [f"--{k}"] + [str(x) for x in v]
        if isinstance(v, dict):
            return [f"--{k}", json.dumps(v)]
        return [f"--{k}", str(v).replace('/DeepEM/', ROOT + '/')]

    argv = ["amp_bench", "--exp_name", "amp_bench", "--test", "--train_ids", "all"]
    argv += [a for k, v in spec.items() for a in fmt(k, v)]
    saved, sys.argv = sys.argv, argv
    try:
        return Options().parse()
    finally:
        sys.argv = saved


def variant(base, width=None, mode=None, channels_last=False):
    opt = copy.copy(base)
    if width is not None:
        opt.width = [width * 2 ** i for i in range(len(base.width))]
    opt.mixed_precision = None if mode in (None, 'fp32') else mode
    opt.channels_last = channels_last
    return opt


def build(opt, seed=0):
    from deepem.train.model import Model
    from deepem.train.utils import get_criteria
    from deepem.utils.py_utils import load_module

    torch.manual_seed(seed)
    net = load_module("model", opt.model).create_model(opt)
    model = Model(net, get_criteria(opt), opt).train().cuda()
    if opt.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    return model


def synthetic_sample(opt, batch, seed, device='cuda'):
    """Input noise plus Voronoi segments; binary heads are random blobs."""
    g = torch.Generator(device=device).manual_seed(seed)
    [(in_key, in_shape)] = opt.in_spec.items()
    sample = {in_key: torch.rand((batch,) + tuple(in_shape), generator=g, device=device)}
    out_shape = next(iter(opt.out_spec.values()))[1:]
    n = math.prod(out_shape)
    coords = torch.stack(torch.meshgrid(*[torch.arange(s, device=device, dtype=torch.float32)
                                          for s in out_shape], indexing='ij'), -1).reshape(n, 3)
    segs = []
    for _ in range(batch):
        seeds = torch.rand((256, 3), generator=g, device=device) * torch.tensor(out_shape, device=device)
        lab = torch.cat([torch.cdist(c, seeds).argmin(1) for c in coords.split(65536)]) + 1
        segs.append(lab.reshape(out_shape).float())
    seg = torch.stack(segs)[:, None]
    for k in opt.out_spec:
        if k in ('embedding', 'long_range') or k.startswith('affinity'):
            sample[k] = seg
        else:
            sample[k] = (torch.rand(seg.shape, generator=g, device=device) > 0.7).float()
        sample[k + '_mask'] = torch.ones_like(seg)
    if 'embedding' in opt.out_spec:
        sample['embedding_split'] = seg
    return sample


def step(model, opt, sample, optimizer):
    for p in model.parameters():
        p.grad = None
    losses, nmasks, preds = model(sample)
    total = sum(w * losses[k].mean() for k, w in opt.loss_weight.items())
    model.precision.backward_step(total, optimizer)
    return {k: v.mean().item() for k, v in losses.items()}


def make_optimizer(opt, model):
    params = dict(opt.optim_params)
    if getattr(opt, 'fused_optim', False):
        params['fused'] = True
    return getattr(torch.optim, opt.optim)(model.parameters(), **params)


def grads_and_losses(opt, sample, seed=0):
    model = build(opt, seed)
    losses, _, _ = model(sample)
    total = sum(w * losses[k].mean() for k, w in opt.loss_weight.items())
    # fp16 trains with a loss scale; use the scaler's initial one here too.
    scale = 2.0 ** 16 if opt.mixed_precision == 'fp16' else 1.0
    (total * scale).backward()
    g = torch.cat([p.grad.flatten().float() for p in model.parameters() if p.grad is not None]) / scale
    return g, {k: v.mean().item() for k, v in losses.items()}


def parity(base, modes, width):
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    sample = synthetic_sample(base, 1, seed=1)
    ref_opt = variant(base, width, 'fp32')
    g_ref, l_ref = grads_and_losses(ref_opt, sample)
    out = []
    for mode in modes:
        if mode == 'fp32':
            continue
        torch.manual_seed(5)
        x = sample['input']
        noisy = dict(sample, input=x * (1 + EPS[mode] * torch.randn_like(x)))
        g_floor, _ = grads_and_losses(ref_opt, noisy)
        g, l = grads_and_losses(variant(base, width, mode), sample)
        out.append({
            'mode': mode, 'width': width,
            'loss_rel_err': {k: abs(l[k] - l_ref[k]) / max(abs(l_ref[k]), 1e-12) for k in l_ref},
            'grad_cos': cos(g_ref, g), 'grad_cos_floor': cos(g_ref, g_floor),
            'finite': bool(torch.isfinite(g).all()),
        })
        r = out[-1]
        print(f"[parity] w{width} {mode}: grad cos {r['grad_cos']:.4f} (fp32 floor at eps "
              f"{EPS[mode]:.1e}: {r['grad_cos_floor']:.4f}), max loss rel err "
              f"{max(r['loss_rel_err'].values()):.2e}, finite={r['finite']}")
    return out


def trajectory(base, modes, width, steps, pool=8):
    samples = [synthetic_sample(base, 1, seed=100 + i) for i in range(pool)]
    runs = {}
    for name in ['fp32', 'fp32_repeat'] + [m for m in modes if m != 'fp32']:
        opt = variant(base, width, name.split('_')[0])
        model = build(opt, seed=0)
        optimizer = make_optimizer(opt, model)
        curve = []
        for i in range(steps):
            losses = step(model, opt, samples[i % pool], optimizer)
            curve.append(sum(losses.values()))
        runs[name] = curve
        print(f"[trajectory] w{width} {name}: first {curve[0]:.4f} last-20 mean "
              f"{statistics.mean(curve[-20:]):.4f}")
        del model, optimizer
        torch.cuda.empty_cache()

    def gap(a, b):  # mean relative gap over the last quarter
        q = len(a) // 4
        return statistics.mean(abs(x - y) / abs(x) for x, y in zip(a[-q:], b[-q:]))

    ref = runs['fp32']
    return {'width': width, 'steps': steps, 'curves': runs,
            'gap_vs_fp32': {k: gap(ref, v) for k, v in runs.items() if k != 'fp32'}}


def bench_one(base, width, mode, batch, channels_last, warmup, iters):
    opt = variant(base, width, mode, channels_last)
    rec = dict(width=width, mode=mode, batch=batch, channels_last=channels_last)
    model = optimizer = None
    try:
        model = build(opt)
        optimizer = make_optimizer(opt, model)
        sample = synthetic_sample(opt, batch, seed=1)
        for _ in range(warmup):
            step(model, opt, sample, optimizer)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            step(model, opt, sample, optimizer)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / 1000.0)
        rec.update(s_per_iter=statistics.median(times),
                   s_per_sample=statistics.median(times) / batch,
                   peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                   params_m=sum(p.numel() for p in model.parameters()) / 1e6)
    except torch.cuda.OutOfMemoryError:
        rec['oom'] = True
    finally:
        del model, optimizer
        torch.cuda.empty_cache()
    msg = ("OOM" if rec.get('oom') else
           f"{rec['s_per_iter']:.3f} s/iter  {rec['peak_gib']:.2f} GiB  {rec['params_m']:.1f}M params")
    print(f"[bench] w{width} {mode:4} bs{batch} cl={int(channels_last)}: {msg}", flush=True)
    return rec


def markdown(res):
    lines = [f"# AMP bench: {res['device']}, torch {res['torch']}, cuDNN {res['cudnn']}", ""]
    if res.get('parity'):
        lines += ["## Parity (one step, same weights and input)", "",
                  "| width | mode | grad cos | fp32 floor | max loss rel err | finite |", "|---|---|---|---|---|---|"]
        for r in res['parity']:
            lines.append(f"| {r['width']} | {r['mode']} | {r['grad_cos']:.4f} | {r['grad_cos_floor']:.4f} | "
                         f"{max(r['loss_rel_err'].values()):.2e} | {r['finite']} |")
        lines.append("")
    if res.get('trajectory'):
        t = res['trajectory']
        lines += [f"## Trajectory ({t['steps']} steps, width {t['width']})", "",
                  "| run | mean rel. gap to fp32, last quarter |", "|---|---|"]
        lines += [f"| {k} | {v:.2e} |" for k, v in t['gap_vs_fp32'].items()]
        lines.append("")
    if res.get('bench'):
        lines += ["## Speed and memory (synthetic data, no loader)", "",
                  "| width | mode | batch | channels_last | s/iter | s/sample | peak GiB |", "|---|---|---|---|---|---|---|"]
        for r in res['bench']:
            if r.get('oom'):
                lines.append(f"| {r['width']} | {r['mode']} | {r['batch']} | {r['channels_last']} | OOM | | |")
            else:
                lines.append(f"| {r['width']} | {r['mode']} | {r['batch']} | {r['channels_last']} | "
                             f"{r['s_per_iter']:.3f} | {r['s_per_sample']:.3f} | {r['peak_gib']:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument('--spec', required=True, help='Seuron training spec JSON')
    ap.add_argument('--out', default='amp_bench_out')
    ap.add_argument('--parity', action='store_true')
    ap.add_argument('--trajectory', type=int, default=0, metavar='STEPS')
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--widths', type=int, nargs='+', default=[32, 48, 64])
    ap.add_argument('--modes', nargs='+', default=['fp32', 'bf16'], choices=['fp32', 'bf16', 'fp16'])
    ap.add_argument('--batch', type=int, nargs='+', default=[1, 2, 4])
    ap.add_argument('--channels_last', nargs='+', default=['off', 'on'], choices=['off', 'on'])
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=20)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "amp_bench needs a GPU"

    torch.backends.cudnn.benchmark = True
    base = spec_opt(args.spec)
    res = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
           'cudnn': torch.backends.cudnn.version(), 'spec': os.path.abspath(args.spec),
           'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    w0 = base.width[0]
    if args.parity:
        res['parity'] = [r for w in args.widths for r in parity(base, args.modes, w)]
    if args.trajectory:
        res['trajectory'] = trajectory(base, args.modes, w0, args.trajectory)
    if args.bench:
        res['bench'] = [bench_one(base, w, m, b, cl == 'on', args.warmup, args.iters)
                        for w in args.widths for m in args.modes
                        for b in args.batch for cl in args.channels_last]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(args.out, 'results.md'), 'w') as f:
        f.write(markdown(res))
    print(f"wrote {args.out}/results.json and results.md")


if __name__ == '__main__':
    main()

"""Activation-function efficiency benchmark.

This is the benchmark behind the paper's efficiency table.  It measures two
things:

  part 1  raw activation cost on a flat vector of float32 elements: forward and
          backward timed separately with CUDA events, `trials` repetitions.
          Paper protocol: 6.7e7 elements, 100 trials.
  part 2  a real training step: ResNet-18 on CIFAR-10, AdamW (lr 1e-3, wd 0.1),
          batch 512, `steps` timed steps x `reps` independent repeats; s/epoch is
          projected with 50000/512 = 97.6 steps per epoch and carries the same
          relative uncertainty as ms/step (the across-repeat sd).

Arms: GELU, Swish (SiLU), Mish, GPS (2 parameters), mixture (3), free mixture (9).
Every trainable form runs its Triton kernel, so the numbers include the kernel
launch and autotune overhead that the paper's numbers include.

Usage
-----
    ./run.sh                          # full paper protocol, writes results.json
    python bench_efficiency.py --skip-train        # part 1 only
    python bench_efficiency.py --n 1000000 --trials 5   # quick smoke test

Run it on an otherwise idle GPU: a single run already saturates the device, and
any concurrent process corrupts both parts.  The script warns if it sees other
compute processes on the GPU.
"""
import argparse
import json
import os
import platform
import statistics as st
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, 'lib'))

import gps_lab        # noqa: E402,F401  (activation registry + Triton kernels)
import gps_choice     # noqa: E402,F401  (registers the mixture forms)
from gps_lab import ActivationRegistry, ArchitectureRegistry, DatasetRegistry  # noqa: E402

# (label, registry name, kwargs) - identical to the paper's benchmark
ARMS = [
    ('GELU',          'gelu',      {}),
    ('Swish',         'silu',      {}),
    ('Mish',          'mish',      {}),
    ('GPS (2p)',      'pdfv4',     {'init': 'gelu'}),
    ('mixture (3p)',  'mixchoice', {'mode': 'blend', 'init': 'minimax'}),
    ('free mix (9p)', 'mixfree',   {'init': 'minimax'}),
]

DEFAULTS = dict(n=67_000_000, trials=100, batch=512, steps=30, reps=3)

# activation parameters per site (per layer) for each trainable form
PER_LAYER = {'pdfv4': 2, 'mixchoice': 3, 'mixfree': 9}

# where an existing CIFAR-10 may already live; searched in this order before
# falling back to `requested` (which makes torchvision download it)
CIFAR_CANDIDATES = [
    os.environ.get('CIFAR10_DIR', ''),
    os.path.join(HERE, 'data'),
    os.path.abspath(os.path.join(HERE, os.pardir, 'data')),
]


def resolve_data_dir(requested):
    """Prefer a directory that already holds cifar-10-batches-py."""
    marker = 'cifar-10-batches-py'
    if os.path.isdir(os.path.join(requested, marker)):
        return requested
    for cand in CIFAR_CANDIDATES:
        if cand and os.path.isdir(os.path.join(cand, marker)):
            print(f'using the CIFAR-10 batches already present in {cand}')
            return cand
    print(f'no existing CIFAR-10 found; {requested} will be populated by the '
          f'torchvision download (or pass --data-dir, or --skip-train)')
    return requested
STEPS_PER_EPOCH = 50_000 / 512          # 97.6, as used in the paper


def fmt_sd(x, nd=3):
    """Format an uncertainty so a tight run cannot print as '0.000'."""
    return f'{x:.{nd}f}' if x >= 5 * 10 ** (-(nd + 1)) else f'<{10 ** -nd:.{nd}f}'


def _act(name, kw):
    """Out-of-place instances: the registry's silu/mish are in-place, which is
    invalid on a leaf input that requires grad."""
    if name == 'silu':
        return torch.nn.SiLU().cuda()
    if name == 'mish':
        return torch.nn.Mish().cuda()
    return ActivationRegistry.get(name, **kw).cuda()


def env_info():
    info = {
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    try:
        import triton
        info['triton'] = triton.__version__
    except Exception:
        info['triton'] = 'not importable'
    for field in ('driver_version', 'clocks.sm', 'clocks.max.sm', 'power.draw',
                  'power.limit', 'temperature.gpu', 'persistence_mode'):
        try:
            out = subprocess.run(
                ['nvidia-smi', f'--query-gpu={field}', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10).stdout.strip()
            if out:
                info[field] = out.splitlines()[0]
        except Exception:
            pass
    return info


def busy_check():
    """Warn (do not fail) if other processes are using the GPU."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                              '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return
    mine = {os.getpid(), os.getppid()}
    others = [ln for ln in out.splitlines()
              if ln.strip() and int(ln.split(',')[0]) not in mine]
    if others:
        print('WARNING: other compute processes are on the GPU; timings will be '
              'contaminated:\n  ' + '\n  '.join(others))


def micro(n, trials):
    print(f'=== part 1: raw activation, N = {n:.1e} float32, {trials} trials')
    print(f'{"activation":16s} {"fwd (ms)":>16s} {"bwd (ms)":>16s} {"fwd+bwd":>10s} '
          f'{"vs GELU":>8s} {"M elem/s":>9s} {"peak GB":>8s}')
    print('(± is the standard error of the mean over the trials; the per-trial sd is in results.json)')
    x = torch.randn(n, device='cuda', requires_grad=True)
    base = None
    rows = []
    for label, name, kw in ARMS:
        act = _act(name, kw)
        for _ in range(3):                                  # warmup + autotune
            y = act(x)
            torch.autograd.grad(y, x, torch.ones_like(y), retain_graph=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(trials)]
        fwd = []
        for s, e in ev:
            s.record(); y = act(x); e.record()
        torch.cuda.synchronize()
        fwd = [s.elapsed_time(e) for s, e in ev]
        go = torch.ones_like(y)
        both = []
        for s, e in ev:
            s.record()
            y = act(x)
            torch.autograd.grad(y, x, go, retain_graph=False)
            e.record()
        torch.cuda.synchronize()
        both = [s.elapsed_time(e) for s, e in ev]
        peak = torch.cuda.max_memory_allocated() / 2**30
        f, b = st.mean(fwd), st.mean(both) - st.mean(fwd)
        f_sd, b_sd = st.stdev(fwd), st.stdev(both)
        if base is None:
            base = st.mean(both)
        rows.append(dict(arm=label, fwd_ms=f, fwd_sd=f_sd, fwd_sem=f_sd / trials ** 0.5,
                         bwd_ms=b, bwd_sd=b_sd, bwd_sem=b_sd / trials ** 0.5,
                         fwd_bwd_ms=st.mean(both), vs_gelu=st.mean(both) / base,
                         m_elem_s=n / st.mean(both) / 1e3, peak_gb=peak))
        print(f'{label:16s} {f:8.4f}+-{f_sd:6.4f} {b:8.4f}+-{b_sd:6.4f} '
              f'{st.mean(both):10.4f} {st.mean(both) / base:8.4f} '
              f'{n / st.mean(both) / 1e3:9.1f} {peak:8.2f}')
        del act
        torch.cuda.empty_cache()
    del x
    torch.cuda.empty_cache()
    return rows


def training(batch, steps, reps, data_dir):
    print(f'\n=== part 2: ResNet-18 / CIFAR-10, AdamW, batch {batch}, '
          f'{steps} timed steps x {reps} repeats')
    data_dir = resolve_data_dir(data_dir)
    try:
        tr, _, in_ch, n_cls = DatasetRegistry.get('cifar10', batch_size=batch,
                                                  data_dir=data_dir, augmentation='none')
    except Exception as exc:                                # no dataset, no network
        print(f'skipped: could not build the CIFAR-10 loader ({type(exc).__name__}: {exc})')
        print('put the CIFAR-10 python batches in ./data or allow the download, '
              'or run with --skip-train')
        return []
    print(f'{"activation":16s} {"ms/step":>16s} {"s/epoch":>15s} {"steps/s":>9s} '
          f'{"vs GELU":>8s}   activation parameters (per layer)')
    print('(s/epoch is projected from ms/step at 97.6 steps/epoch; its uncertainty '
          'is the across-repeat sd of ms/step, scaled the same way)')
    base = None
    out = []
    for label, name, kw in ARMS:
        model = ArchitectureRegistry.get('resnet18', num_classes=n_cls,
                                         activation=name, in_channels=in_ch,
                                         act_kwargs=kw).cuda()
        n_act = sum(p.numel() for n_, p in model.named_parameters() if 'activation' in n_)
        per_layer = PER_LAYER.get(name, 0)
        n_sites = n_act // per_layer if per_layer else 0
        params_txt = (f'{n_act} ({per_layer}/layer)' if per_layer else '0')
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        it = iter(tr)

        def next_batch():
            nonlocal it
            try:
                return next(it)
            except StopIteration:                           # re-shuffle if exhausted
                it = iter(tr)
                return next(it)

        for _ in range(3):                                  # warmup
            xb, yb = next_batch()
            xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
        torch.cuda.synchronize()
        run = []
        for _ in range(reps):                               # independent repeats
            ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(steps)]
            for s_, e_ in ev:
                xb, yb = next_batch()
                xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
                s_.record()
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(xb), yb)
                loss.backward(); opt.step()
                e_.record()
            torch.cuda.synchronize()
            run.append(st.mean(s_.elapsed_time(e_) for s_, e_ in ev))
        ms, ms_sd = st.mean(run), st.stdev(run)
        if base is None:
            base = ms
        ep = ms * STEPS_PER_EPOCH / 1000
        ep_sd = ms_sd * STEPS_PER_EPOCH / 1000        # same relative uncertainty
        out.append(dict(arm=label, ms_step=ms, ms_step_sd=ms_sd,
                        s_epoch=ep, s_epoch_sd=ep_sd, steps_s=1000 / ms,
                        vs_gelu=ms / base, act_params=n_act,
                        params_per_layer=per_layer, act_sites=n_sites))
        print(f'{label:16s} {ms:8.2f}+-{fmt_sd(ms_sd):>7s} {ep:9.2f}+-{fmt_sd(ep_sd):>7s} '
              f'{1000 / ms:9.2f} {ms / base:8.4f}   {params_txt}')
        del model, opt
        torch.cuda.empty_cache()
    return out


def report(info, m, t, cfg):
    lines = ['# Activation efficiency benchmark', '',
             f'- GPU: {info["gpu"]}  (SM clock {info.get("clocks.sm", "?")} / '
             f'max {info.get("clocks.max.sm", "?")}, power {info.get("power.draw", "?")} '
             f'of {info.get("power.limit", "?")})',
             f'- torch {info["torch"]}, triton {info.get("triton", "?")}, '
             f'CUDA {info["cuda"]}, driver {info.get("driver_version", "?")}',
             f'- {info["timestamp"]}', '']
    if m:
        lines += [f'## Part 1 - raw activation ({cfg["n"]:.1e} float32, '
                  f'{cfg["trials"]} trials)', '',
                  '| activation | forward (ms) | backward (ms) | fwd+bwd (ms) | vs GELU | M elem/s | peak GB |',
                  '|---|---|---|---|---|---|---|']
        for r in m:
            lines.append(f'| {r["arm"]} | {r["fwd_ms"]:.3f} ± {fmt_sd(r["fwd_sem"])} | '
                         f'{r["bwd_ms"]:.3f} ± {fmt_sd(r["bwd_sem"])} | {r["fwd_bwd_ms"]:.3f} | '
                         f'{r["vs_gelu"]:.3f} | {r["m_elem_s"]:.1f} | {r["peak_gb"]:.2f} |')
        lines.append('')
    if t:
        lines += [f'## Part 2 - ResNet-18 / CIFAR-10 training step '
                  f'(batch {cfg["batch"]}, {cfg["steps"]} steps x {cfg["reps"]} repeats)', '',
                  '| activation | ms/step | s/epoch | steps/s | vs GELU | act params |',
                  '|---|---|---|---|---|---|']
        for r in t:
            ep = r.get('s_epoch_sd')
            ep_txt = (f'{r["s_epoch"]:.2f} ± {fmt_sd(ep)}' if ep is not None
                      else f'{r["s_epoch"]:.2f}')
            lines.append(f'| {r["arm"]} | {r["ms_step"]:.2f} ± {r["ms_step_sd"]:.2f} | '
                         f'{ep_txt} | {r["steps_s"]:.1f} | {r["vs_gelu"]:.3f} | '
                         f'{r["act_params"]} |')
        lines += ['', 'Raw columns: ± is the standard error of the mean over the trials '
                  '(per-trial sd in results.json). Training columns: ± is the sd across the '
                  'independent repeats; s/epoch is projected from ms/step at 97.6 steps per '
                  'epoch and carries the same relative uncertainty.']
        lines.append('')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=DEFAULTS['n'], help='part 1 vector length')
    ap.add_argument('--trials', type=int, default=DEFAULTS['trials'], help='part 1 trials')
    ap.add_argument('--batch', type=int, default=DEFAULTS['batch'])
    ap.add_argument('--steps', type=int, default=DEFAULTS['steps'], help='timed steps per repeat')
    ap.add_argument('--reps', type=int, default=DEFAULTS['reps'], help='independent repeats')
    ap.add_argument('--data-dir', default=os.path.join(HERE, 'data'))
    ap.add_argument('--skip-train', action='store_true', help='part 1 only')
    ap.add_argument('--train-only', action='store_true', help='part 2 only')
    ap.add_argument('--json', default=os.path.join(HERE, 'results.json'))
    ap.add_argument('--markdown', default=os.path.join(HERE, 'results.md'))
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA device required')
    print(f'gpu: {torch.cuda.get_device_name(0)} | torch {torch.__version__}')
    busy_check()
    t0 = time.time()
    m = [] if a.train_only else micro(a.n, a.trials)
    t = [] if a.skip_train else training(a.batch, a.steps, a.reps, a.data_dir)
    info = env_info()
    payload = dict(config=vars(a), env=info, micro=m, training=t)
    with open(a.json, 'w') as f:
        json.dump(payload, f, indent=2)
    md = report(info, m, t, vars(a))
    with open(a.markdown, 'w') as f:
        f.write(md + '\n')
    print('\n' + md)
    print(f'wrote {a.json} and {a.markdown}   (total {(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()

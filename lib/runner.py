"""The machinery the test scripts drive: config assembly, the resumable job
loop, and the results export.

No experiment choice lives here.  What to run is spelled out in the CONFIG
block at the top of each script in tests/; the training recipe, the arm tables
and the benchmark cells live in recipe.py / tests/main_suite.py.  This module
turns those into ExpConfig objects and runs them.

Two suites share the config assembly:

  run_vision()     the vision cells -> results.{json,md,csv}
  run_nonvision()  the text/regression cells -> results_nonvision.{json,md,csv}
                   (parses the tier suffix out of artifact names and reports
                    California housing as -MSE so that higher is always better)

Both are resumable: a job whose <out>/<name>/summary.json exists is skipped, so
a run can be stopped and restarted at any time.  Jobs that fail are reported
and skipped over, never aborting the suite.
"""
import argparse
import csv
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

# make recipe.py (parent dir) and the sibling lib modules importable no matter
# who imports us first
_HERE = Path(__file__).resolve()
for _p in (str(_HERE.parents[1]), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

import gps_lab                                   # noqa: F401  (Trainer, ExpConfig)
import gps_choice                                # noqa: F401  (mixchoice/mixfree)
import models                                    # noqa: F401  (charlm/regmlp + text datasets)
import trainable_acts                           # noqa: F401  (beta-swish & co.)
from gps_lab import ExpConfig, run_experiment

import recipe

AUG = {'vision': recipe.AUG_VISION, 'none': recipe.AUG_NONE}

# artifact-name grammar of the non-vision cells; export_nonvision recovers
# (arm, cell, seed, tier) from a directory name with it
PAT = re.compile(r'^(?P<arm>.+)_(?P<cell>wikitext_glu|openorca_glu|wikitext|openorca|calhousing)'
                 r'_s(?P<seed>\d+)(?P<tier>_m50|_big)?$')


# ------------------------------------------------------------- config
def make_cfg(cell, arm, seed, arm_cfg=None, epochs=None, batch=None,
             share=None, mult=None, data_dir=None):
    """recipe.CELLS[cell] x the arm x the run knobs -> ExpConfig.

    Layering matches the original runners: recipe, then the cell, then the
    arm, then any explicit run knob.  `arm_cfg` is the arm's own dict (the
    main suite's arm table carries activation + act_lr_mult + share); without
    it the arm name is taken to be the activation module itself, which is how
    the baseline suite builds beta_swish, pau, ... .
    """
    row = recipe.CELLS[cell]
    merged = {**recipe.RECIPE, **recipe.ACT_PARAMS, **AUG[row['aug']],
              'arch': row['arch'], 'dataset': row['dataset'],
              'epochs': row['epochs'] if not epochs else epochs,
              'batch_size': row['batch'] if batch is None else batch,
              'data_dir': recipe.RECIPE['data_dir'] if data_dir is None else data_dir}
    if row.get('eval_metric'):
        merged['eval_metric'] = row['eval_metric']
    merged.update(arm_cfg or {'activation': arm, 'activation_kwargs': {}})
    if share is not None:
        merged['share'] = share
    if mult is not None:
        merged['act_lr_mult'] = mult
    stem, suffix = row.get('stem', cell), row.get('suffix', '')
    return ExpConfig(name=f'{arm}_{stem}_s{seed}{suffix}', seed=seed, **merged)


def env_block(args):
    block = {'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
             'torch': torch.__version__, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
             'share': args.share, 'act_lr_mult': args.mult, 'epochs': args.epochs}
    for k in ('cuda', 'triton'):
        try:
            block[k] = torch.version.cuda if k == 'cuda' else __import__('triton').__version__
        except Exception:
            pass
    try:
        block['git'] = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                      capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    return block


# ------------------------------------------------------------- export
def export_vision(out_dir, args, jobs_done, jobs_total, t0):
    """Write results.json / results.md / results.csv from whatever exists."""
    out = Path(out_dir)
    runs = []
    for d in sorted(out.glob('*/summary.json')):
        try:
            s = json.load(open(d))
            c = json.load(open(d.parent / 'config.json'))
        except Exception:
            continue
        runs.append({'cell': c.get('dataset'), 'arch': c.get('arch'), 'arm': c['activation'],
                     'seed': c.get('seed'), 'acc': s['final_test_acc'],
                     'epochs': c.get('epochs'), 'batch': c.get('batch_size'),
                     'share': c.get('share'), 'mult': c.get('act_lr_mult'),
                     'act_params': s.get('act_params')})
    payload = {'env': env_block(args), 'runs': runs,
               'jobs_done': jobs_done, 'jobs_total': jobs_total,
               'elapsed_min': round((time.time() - t0) / 60, 1)}
    (out / 'results.json').write_text(json.dumps(payload, indent=2))

    # group by (arch, cell, arm)
    groups = {}
    for r in runs:
        groups.setdefault((r['arch'], r['cell'], r['arm']), []).append(r['acc'])
    rows = []
    for (arch, cell, arm), vals in sorted(groups.items()):
        rows.append({'arch': arch, 'cell': cell, 'arm': arm, 'n': len(vals),
                     'mean': st.mean(vals), 'sd': st.stdev(vals) if len(vals) > 1 else 0.0,
                     'values': sorted(round(v, 3) for v in vals)})
    with open(out / 'results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['arch', 'cell', 'arm', 'n', 'mean', 'sd', 'values'])
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'values': ' '.join(str(v) for v in r['values'])})

    lines = ['# Trainable-activation baselines', '',
             f"- share=`{args.share}`, activation lr = {args.mult}x weight lr, "
             f"{args.epochs} epochs",
             f"- {len(runs)} runs finished of {jobs_total} queued "
             f"({round((time.time() - t0) / 60, 1)} min elapsed)", '']
    if rows:
        lines += ['| arch | cell | arm | n | mean | sd |', '|---|---|---|---|---|---|']
        for r in sorted(rows, key=lambda r: (r['cell'], -r['mean'])):
            lines.append(f"| {r['arch']} | {r['cell']} | {r['arm']} | {r['n']} | "
                         f"{r['mean']:.2f} | {r['sd']:.2f} |")
    (out / 'results.md').write_text('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines[-len(rows) - 5:]))
    print(f"\nwrote {out}/results.json, results.md, results.csv")


def export_nonvision(out_dir, args, t0):
    """Write results_nonvision.{json,md,csv} from whatever exists."""
    out = Path(out_dir)
    runs = []
    for d in sorted(out.glob('*/summary.json')):
        m = PAT.match(d.parent.name)
        if not m:
            continue
        try:
            s = json.load(open(d))
            c = json.load(open(d.parent / 'config.json'))
        except Exception:
            continue
        cell = m['cell'] + (m['tier'] or '')
        acc = s['final_test_acc']
        runs.append({'cell': cell, 'arm': m['arm'], 'seed': int(m['seed']),
                     'value': -acc if m['cell'] == 'calhousing' else acc,   # -MSE: higher better
                     'epochs': c.get('epochs'), 'batch': c.get('batch_size'),
                     'share': c.get('share'), 'mult': c.get('act_lr_mult'),
                     'act_params': s.get('act_params')})
    env = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
           'share': args.share, 'mult': args.mult,
           'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    try:
        env['git'] = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                    capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    (out / 'results_nonvision.json').write_text(
        json.dumps({'env': env, 'runs': runs, 'elapsed_min': round((time.time() - t0) / 60, 1)},
                   indent=2))

    groups = {}
    for r in runs:
        groups.setdefault((r['cell'], r['arm']), []).append(r['value'])
    rows = [{'cell': c, 'arm': a, 'n': len(v), 'mean': st.mean(v),
             'sd': st.stdev(v) if len(v) > 1 else 0.0,
             'values': sorted(round(x, 3) for x in v)}
            for (c, a), v in sorted(groups.items())]
    with open(out / 'results_nonvision.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['cell', 'arm', 'n', 'mean', 'sd', 'values'])
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'values': ' '.join(str(x) for x in r['values'])})
    lines = ['# Trainable-activation baselines, non-vision cells', '',
             f"- share=`{args.share}`, activation lr = {args.mult}x weight lr",
             f"- {len(runs)} runs, {round((time.time() - t0) / 60, 1)} min elapsed", '',
             '| cell | arm | n | mean | sd | values |', '|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['cell']} | {r['arm']} | {r['n']} | {r['mean']:.2f} | {r['sd']:.2f} | "
                     f"{' '.join(str(x) for x in r['values'])} |")
    (out / 'results_nonvision.md').write_text('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines[4:]))
    print(f'\nwrote {out}/results_nonvision.{{json,md,csv}}')


# ------------------------------------------------------------- the loop
def _parser(description, cfg):
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', default=','.join(cfg['arms']),
                    help=f"comma list, default {','.join(cfg['arms'])}; "
                         f"available: {','.join(trainable_acts.ARMS)}")
    ap.add_argument('--seeds', default=','.join(map(str, cfg['seeds'])))
    ap.add_argument('--cells', default=','.join(cfg['cells']),
                    help='comma list of recipe.CELLS names')
    ap.add_argument('--epochs', type=int, default=cfg.get('epochs', 0),
                    help='override the per-cell epoch count (0 = use the cell matrix)')
    ap.add_argument('--share', default=cfg['share'], choices=['none', 'global', 'block'])
    ap.add_argument('--mult', type=float, default=cfg['mult'])
    ap.add_argument('--batch', type=int, default=None,
                    help='override the batch of every cell (default: the CONFIG block)')
    ap.add_argument('--out', default=cfg['out'])
    ap.add_argument('--data-dir', default=recipe.RECIPE['data_dir'])
    ap.add_argument('--gpu', default=None)
    ap.add_argument('--dry-run', action='store_true', help='list the jobs and exit')
    return ap


def _expand(args, batch_spec=None):
    """All (cfg, corpus) pairs of the requested grid, minus the finished ones.

    batch_spec is the CONFIG 'batch' of the calling test script: either one
    int for every cell, or {cell: batch} where a cell differs from the matrix.
    --batch on the command line overrides both.
    """
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    cells = [c.strip() for c in args.cells.split(',') if c.strip()]
    jobs = []
    for cell in cells:
        row = recipe.CELLS[cell]
        if args.batch is not None:
            batch = args.batch
        elif isinstance(batch_spec, dict):
            batch = batch_spec.get(cell)
        else:
            batch = batch_spec
        for arm in arms:
            for seed in seeds:
                cfg = make_cfg(cell, arm, seed, epochs=args.epochs, batch=batch,
                               share=args.share, mult=args.mult, data_dir=args.data_dir)
                done = (Path(args.out) / cfg.name / 'summary.json').exists()
                jobs.append((cfg, row.get('corpus'), done))
    todo = [(c, k) for c, k, d in jobs if not d]
    print(f'{len(jobs)} jobs, {len(jobs) - len(todo)} already finished, {len(todo)} to run')
    print(f"share={args.share}  mult={args.mult}  "
          f"epochs={args.epochs or 'per-cell'}  cells={cells}  arms={arms}  seeds={seeds}")
    return todo


def _loop(todo, args, export):
    """Run every job; export every 5 jobs and at the end.  `export(i, total)`."""
    t0, times = time.time(), []
    for i, (job, corpus) in enumerate(todo, 1):
        eta = f'  ETA {st.mean(times) * (len(todo) - i + 1) / 60:.0f} min' if times else ''
        print(f'--- [{i}/{len(todo)}] {job.name}  '
              f'(elapsed {(time.time() - t0) / 60:.0f} min{eta})', flush=True)
        if corpus:
            models.CORPUS_MAX = corpus        # corpus tier of the text cells
        start = time.time()
        try:
            run_experiment(job, out_dir=args.out)
            times.append(time.time() - start)
        except Exception as exc:              # keep going, record the failure
            print(f'!!! {job.name} failed: {type(exc).__name__}: {exc}', flush=True)
        if i % 5 == 0 or i == len(todo):
            export(i, len(todo))
    print(f'\nall done in {(time.time() - t0) / 60:.0f} min')


def run_vision(cfg, description=''):
    """The vision baselines: one (cell, arm, seed) per job, 5-job exports."""
    args = _parser(description, cfg).parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    Path(args.out).mkdir(parents=True, exist_ok=True)
    todo = _expand(args, cfg.get('batch'))
    if args.dry_run:
        for c, _ in todo[:10]:
            print('  would run', c.name, f'({c.arch}/{c.dataset}, batch {c.batch_size})')
        print(f'  ... {max(0, len(todo) - 10)} more')
        return
    t0 = time.time()
    _loop(todo, args,
          lambda i, total: export_vision(args.out, args, total - i, total, t0))
    export_vision(args.out, args, len(todo), len(todo), t0)


def run_nonvision(cfg, description=''):
    """The text/regression baselines: per-cell epochs and corpus tiers."""
    args = _parser(description, cfg).parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    Path(args.out).mkdir(parents=True, exist_ok=True)
    todo = _expand(args, cfg.get('batch'))
    if args.dry_run:
        for c, corpus in todo:
            print(f'  would run {c.name} ({c.arch}/{c.dataset}, {c.epochs} epochs, '
                  f'batch {c.batch_size}, corpus {(corpus or 0) / 1e6:.0f} MB)')
        return
    t0 = time.time()
    _loop(todo, args, lambda i, total: export_nonvision(args.out, args, t0))
    export_nonvision(args.out, args, t0)

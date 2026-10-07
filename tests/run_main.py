"""The main suite: every arm of tab:accs, run as a grid over (cell, arm, seed).

    python run_main.py --cells rn18c100 --arms mix_persite,static_gelu --seeds 1,2,3,4,5
    python run_main.py --cells wikitext_m50 --arms naive2p_mauto --seeds 1 --dry-run

Each job trains one network and writes <out>/<arm>_<cell>_s<seed>/ (config.json,
history.json, summary.json, model.pt); a job whose summary.json already exists
is skipped, so the grid can be stopped and restarted at any time.  The cells
(architecture, dataset, batch, epochs, corpus tier) are recipe.CELLS; the
training recipe is recipe.RECIPE.

The columns of tab:accs come from these arms: static_gelu / static_silu /
static_mish, naive2p_mauto (the 'GPS' column), and mix_persite (the 'Mixed Wave
GPS' column); the beta-Swish column is tests/run_baselines_vision.py.

Everything this suite varies is on this page: ARMS below (which activation each
arm is, how it is initialized, its lr multiplier, its parameter sharing) and
PROBE_MULTIPLIER (the per-cell activation lr multiplier of the naive2p_mauto
arm - in the research repository this was hidden in a GPS_ACT_MULT environment
variable).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models                                    # noqa: F401  registers the datasets
import runner
import recipe
from gps_lab import run_experiment

OUT = 'controls6'

# ------------------------------------------------------------- the arms
# name -> how to build the activation module.  'act_lr_mult' is the activation
# lr as a multiple of the weight lr; 'share' overrides recipe.RECIPE's default
# and selects one parameter set per activation site ('none') vs one for the
# whole model ('global').  The frozen_* arms are the frozen-curve controls
# (act_lr_mult = 0: the curve never moves).
ARMS = {
    'static_gelu': dict(activation='gelu'),
    'static_silu': dict(activation='silu'),
    'static_mish': dict(activation='mish'),
    'mix': dict(activation='mixchoice',
                activation_kwargs={'mode': 'blend', 'init': 'minimax'},
                act_lr_mult=0.5, act_decay_until=1.0),
    # NON-reparameterized control: naive 2-param (a,b) training from the
    # curve-space center (0.221, 0), same multipliers as the study
    'naive2p_m0125': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                          act_lr_mult=0.125, act_decay_until=1.0),
    'naive2p_m25': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                        act_lr_mult=0.25, act_decay_until=1.0),
    # ---- Phase F: mechanism arms ----
    'mixfree': dict(activation='mixfree', activation_kwargs={'init': 'minimax'},
                    act_lr_mult=0.5, act_decay_until=1.0),
    'frozen_textpt': dict(activation='pdfv4', activation_kwargs={'a': 0.9, 'b': -0.07},
                          act_lr_mult=0.0),
    'frozen_gelufit': dict(activation='pdfv4', activation_kwargs={'a': 0.3503, 'b': 0.0},
                           act_lr_mult=0.0),
    'frozen_center': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                          act_lr_mult=0.0),
    'frozen_vispt': dict(activation='pdfv4', activation_kwargs={'a': 2.3, 'b': -0.28},
                         act_lr_mult=0.0),
    'naive2p_textstart': dict(activation='pdfv4', activation_kwargs={'a': 0.9, 'b': -0.07},
                              act_lr_mult=0.5, act_decay_until=1.0),
    'naive2p_persite_m1': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                               share='none', act_lr_mult=1.0, act_decay_until=1.0),
    'naive2p_persite_hi': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                               share='none', act_lr_mult=2.0, act_decay_until=1.0),
    'naive2p_mauto': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                          act_decay_until=1.0),
    'naive2p_tr10': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                         act_lr_mult=1.0, act_decay_until=1.0, act_trust_region=0.1),
    'naive2p_tr': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                       act_lr_mult=1.0, act_decay_until=1.0, act_trust_region=0.05),
    'naive2p_tr_hi': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                          act_lr_mult=1.0, act_decay_until=1.0, act_trust_region=0.15),
    'naive2p_persite': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                            share='none', act_lr_mult=0.5, act_decay_until=1.0),
    'mix_persite': dict(activation='mixchoice',
                        activation_kwargs={'mode': 'blend', 'init': 'minimax'},
                        share='none', act_lr_mult=0.5, act_decay_until=1.0),
    'mix_m0125': dict(activation='mixchoice',
                      activation_kwargs={'mode': 'blend', 'init': 'minimax'},
                      act_lr_mult=0.125, act_decay_until=1.0),
    'mix_m25': dict(activation='mixchoice',
                    activation_kwargs={'mode': 'blend', 'init': 'minimax'},
                    act_lr_mult=0.25, act_decay_until=1.0),
    'naive2p_m005': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                         act_lr_mult=0.05, act_decay_until=1.0),
    'naive2p_m05': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                        act_lr_mult=0.5, act_decay_until=1.0),
    'naive2p_m1': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                       act_lr_mult=1.0, act_decay_until=1.0),
}

ARMS.update({
    'static_silu_gate': dict(activation='silu'),
    'static_gelu_gate': dict(activation='gelu'),
    'static_mish_gate': dict(activation='mish'),
    'mix_gate': dict(activation='mixchoice',
                     activation_kwargs={'mode': 'blend', 'init': 'minimax'},
                     act_lr_mult=0.5, act_decay_until=1.0),
    'naive2p_gate_m05': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                             act_lr_mult=0.5, act_decay_until=1.0),
    'naive2p_gate_m1': dict(activation='pdfv4', activation_kwargs={'a': 0.221, 'b': 0.0},
                            act_lr_mult=1.0, act_decay_until=1.0),
})

# Per-cell activation lr multiplier of naive2p_mauto, the arm that carries the
# paper's only hyperparameter rule: an in-run probe measures the first-epoch
# displacement rate of the activation parameters and sets
#     mult = 0.7 / (rate * (epochs - 1)).
# These are the values the tab:accs runs used.  rn18 (1.25) and every text /
# regression cell are probe-derived; rn18c100 and rn50c100 keep the 0.5 fallback
# (no separate probe run was made for them, and 0.5 is within the seed spread of
# the value a probe would give -- see report/HYPERPARAMS.md 5).  A cell not
# listed falls back to 0.5.
PROBE_MULTIPLIER = {
    'rn18': 1.25, 'rn18c100': 0.5, 'rn50c100': 0.5, 'rn50c100_b128': 0.5,
    'wikitext': 0.55, 'wikitext_m50': 0.31, 'wikitext_big': 0.53,
    'openorca': 0.42, 'openorca_big': 0.54,
    'wikitext_glu': 0.55, 'openorca_glu': 0.68,
    'calhousing': 1.04, 'charlm': 4.0,
}


def build(cell, arm, seed):
    o = dict(ARMS[arm])
    if arm == 'naive2p_mauto':                    # per-cell activation lr multiplier
        o['act_lr_mult'] = PROBE_MULTIPLIER.get(cell, 0.5)
    return runner.make_cfg(cell, arm, seed, arm_cfg=o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cells', required=True,
                    help='comma list of recipe.CELLS names')
    ap.add_argument('--arms', required=True,
                    help=f"comma list of ARMS names; available: {', '.join(sorted(ARMS))}")
    ap.add_argument('--seeds', required=True, help='comma list of integer seeds')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--dry-run', action='store_true', help='list the jobs and exit')
    a = ap.parse_args(argv)

    cells = [c.strip() for c in a.cells.split(',') if c.strip()]
    arms = [x.strip() for x in a.arms.split(',') if x.strip()]
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    unknown = [c for c in cells if c not in recipe.CELLS]
    if unknown:
        raise SystemExit(f"unknown cells {unknown}; available: {', '.join(recipe.CELLS)}")

    jobs = [(c, arm, s) for c in cells for arm in arms for s in seeds]
    todo = [j for j in jobs
            if not (Path(a.out) / build(*j).name / 'summary.json').exists()]
    print(f'{len(jobs)} jobs, {len(jobs) - len(todo)} already finished, {len(todo)} to run')
    print(f'cells={cells}  arms={arms}  seeds={seeds}')
    if a.dry_run:
        for j in todo:
            cfg = build(*j)
            print(f'  would run {cfg.name} ({cfg.arch}/{cfg.dataset}, {cfg.epochs} epochs, '
                  f'batch {cfg.batch_size})')
        return

    t0, times = time.time(), []
    for i, job in enumerate(todo, 1):
        cfg = build(*job)
        row = recipe.CELLS[job[0]]
        if row.get('corpus'):
            models.CORPUS_MAX = row['corpus']   # corpus tier of the text cells
        eta = f'  ETA {sum(times) / len(times) * (len(todo) - i + 1) / 60:.0f} min' if times else ''
        print(f'--- [{i}/{len(todo)}] {cfg.name}  '
              f'(elapsed {(time.time() - t0) / 60:.0f} min{eta})', flush=True)
        start = time.time()
        try:
            run_experiment(cfg, out_dir=a.out)
            times.append(time.time() - start)
        except Exception as exc:
            print(f'!!! {cfg.name} failed: {type(exc).__name__}: {exc}', flush=True)
    print(f'\nall done in {(time.time() - t0) / 60:.0f} min')


if __name__ == '__main__':
    main()

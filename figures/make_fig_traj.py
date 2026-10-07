"""Paper figures: where training moves the two GPS parameters.

Two separate single-panel figures, no titles and no text inside the axes - every
marker is identified in the legend instead.

  fig_traj_single  the shared single-term form.  Because one (a, b) pair is shared
                   by every activation site, a run produces exactly ONE
                   trajectory, so each curve is one dataset and one seed.
  fig_traj_sites   the per-site form, where every activation site has its own
                   pair, so one run produces one trajectory per layer.  One
                   ResNet-18/CIFAR-100 run (seed 3); the 21 sites are colored by
                   relative depth.

Both start at the shared initialization (0.221, 0) - the value a frozen curve
keeps for the whole run - and mark the three static fits.

    python paper/make_fig_traj.py    # -> paper/fig_traj_single.pdf / fig_traj_sites.pdf
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
# The training artifacts (controls6/<arm>_<cell>_s<seed>/history.json) are
# looked up at GPS_ARTIFACTS, or else the nearest ancestor of this file that
# contains a controls6/ directory - the research repository root when this
# folder is copied out of it.
ROOT = Path(os.environ.get('GPS_ARTIFACTS', '') or next(
    (d for d in [HERE, *HERE.parents] if (d / 'controls6').is_dir()), HERE.parent))
FITS = {'GELU': (0.3503, 0.0), 'Swish': (0.1458, 0.0), 'Mish': (0.2196, 0.1671)}
INIT = (0.221, 0.0)

SINGLE = [('WikiText-103', 'naive2p_mauto', 'wikitext', '', '#d62728', '-'),
          ('OpenOrca', 'naive2p_mauto', 'openorca', '', '#2ca02c', '-'),
          ('Cal. housing', 'naive2p_mauto', 'calhousing', '', '#9467bd', '-'),
          ('CIFAR-10', 'naive2p_mauto', 'rn18', '', '#1f77b4', '-'),
          ('CIFAR-100', 'naive2p_mauto', 'rn18c100', '', '#ff7f0e', '-'),
          ('WikiText GLU', 'naive2p_mauto', 'wikitext_glu', '', '#d62728', ':'),
          ('OpenOrca GLU', 'naive2p_mauto', 'openorca_glu', '', '#2ca02c', ':')]
NOT_THIS_TIER = ('_big', '_m50', '_b128', '_b512', '_ms25')
SITE_RUN = ('naive2p_persite', 'rn18c100', '', 3)
SUF = ''                  # filename suffix: '' for the single-column set, '_row' for the row

# the constrained mixture lives in the hull of the three fits; each of its
# weights w_i enters the (a, b) plane through sum_i w_i * (a_i, b_i)
MIX = [('WikiText-103', 'mix', 'wikitext', '', '#d62728'),
       ('OpenOrca', 'mix', 'openorca', '', '#2ca02c'),
       ('Cal. housing', 'mix', 'calhousing', '', '#9467bd'),
       ('CIFAR-10', 'mix', 'rn18', '', '#1f77b4'),
       ('CIFAR-100, site mean', 'mix_persite', 'rn18c100', '', '#ff7f0e')]

# all three panels share the per-site panel's font and marker sizes
SINGLE_STYLE = dict(size=(4.4, 3.5), base=12, legend=7, tick=10, mix_legend=7)
STYLE = SINGLE_STYLE
MARK_END = 12       # trajectory endpoint
MARK_STAR = 130     # static-fit corners
MARK_INIT = 70      # shared initialization
MARK_EDGE = 0.3     # marker outline width


def use_style(style):
    global STYLE
    STYLE = style
    plt.rcParams.update({'font.size': style['base'], 'axes.labelsize': style['base'],
                         'legend.fontsize': style['legend'],
                         'xtick.labelsize': style['tick'], 'ytick.labelsize': style['tick']})


def shared_traj(path):
    h = json.load(open(path))
    return np.array([(list(e['act_params'].values())[0][0],
                      list(e['act_params'].values())[0][1])
                     for e in h if e.get('act_params')])


def site_trajs(path):
    h = json.load(open(path))
    series = {}
    for e in h:
        for k, v in (e.get('act_params') or {}).items():
            series.setdefault(k, []).append((v[0], v[1]))
    return series


def softmax(v):
    v = np.asarray(v, dtype=float)
    e = np.exp(v - v.max())
    return e / e.sum()


def fold(w, steps=80):
    """Least-squares single-term fit to the blended curve (the paper's phi)."""
    x = np.linspace(-8, 8, 4000)
    sp = lambda z: np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
    bump = sum(w[i] * np.exp(np.clip(-FITS[k][0] * x * x - FITS[k][1] * x, -80, 80))
               for i, k in enumerate(('GELU', 'Swish', 'Mish')))
    y = sp(x) - np.log(2.0) * bump
    hull = (np.asarray(w)[:, None] * np.array([FITS['GELU'], FITS['Swish'], FITS['Mish']])).sum(0)
    a, b = float(hull[0]), float(hull[1])
    for _ in range(steps):
        g = np.exp(np.clip(-a * x * x - b * x, -80, 80))
        r = (sp(x) - np.log(2.0) * g) - y
        a = min(1.5, max(0.01, a - 0.05 * np.mean(2 * r * (np.log(2.0) * g * x * x))))
        b = min(1.5, max(-1.5, b - 0.05 * np.mean(2 * r * (np.log(2.0) * g * x))))
    return a, b


def mix_traj(path):
    """Per-epoch folded parameters of a mixture arm (site-mean weights)."""
    out = []
    for e in json.load(open(path)):
        ap = e.get('act_params') or {}
        if not ap:
            continue
        w = np.mean([softmax(v) for v in ap.values()], axis=0)
        out.append(fold(w))
    return np.array(out)


def marker_handles():
    """Legend entries for the fits and the starting point (no text in the axes)."""
    return [Line2D([], [], ls='none', marker='*', ms=11, color='black', label='static fits'),
            Line2D([], [], ls='none', marker='P', ms=8, color='black',
                   label='initialization')]


def panel_letter(ax, letter):
    """A/B/C just outside the top-left corner of the frame."""
    ax.text(0.0, 1.015, letter, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=STYLE['base'], fontweight='bold', zorder=10, clip_on=False)


def finish(fig, ax):
    ax.set_xlabel('$a$')
    ax.set_ylabel('$b$')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.subplots_adjust(top=max(fig.subplotpars.top - 0.03, 0.5))


def main():
    # ---------------- figure 1: shared single-term form ----------------
    fig, ax = plt.subplots(figsize=STYLE['size'])
    seen = set()
    for label, arm, cell, tier, col, ls in SINGLE:
        dirs = sorted((ROOT / 'controls6').glob(f'{arm}_{cell}_s*{tier}'))
        if not tier:
            dirs = [d for d in dirs if not d.name.endswith(NOT_THIS_TIER)]
        for d in dirs:
            f = d / 'history.json'
            if not f.exists():
                continue
            tr = np.vstack([INIT, shared_traj(f)])
            ax.plot(tr[:, 0], tr[:, 1], color=col, alpha=0.55, lw=1.6, ls=ls)
            ax.scatter(tr[-1, 0], tr[-1, 1], color=col, s=MARK_END, zorder=6,
                       edgecolors='black', linewidths=MARK_EDGE)
            seen.add((label, col, ls))
    ax.scatter([INIT[0]], [INIT[1]], marker='P', s=MARK_INIT, c='black', zorder=8)
    for _, (a, b) in FITS.items():
        ax.scatter([a], [b], marker='*', s=MARK_STAR, c='black', zorder=8)
    handles = [Line2D([], [], color=c, ls=ls, lw=2, label=lbl)
               for lbl, c, ls in sorted(seen)] + marker_handles()
    ax.legend(handles=handles, loc='best', ncol=2, fontsize=STYLE['legend'], framealpha=0.85,
              handlelength=1.4, columnspacing=0.8, borderpad=0.3, labelspacing=0.25)
    panel_letter(ax, 'A')
    finish(fig, ax)
    for ext in ('pdf', 'png'):
        fig.savefig(HERE / f'fig_traj_single{SUF}.{ext}', dpi=300)
    plt.close(fig)

    # ---------------- figure 2: per-site form (one trajectory per layer) ------
    arm, cell, tier, seed = SITE_RUN
    series = site_trajs(ROOT / 'controls6' / f'{arm}_{cell}_s{seed}{tier}' / 'history.json')
    keys = list(series)
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=STYLE['size'])
    for i, k in enumerate(keys):
        tr = np.vstack([INIT, series[k]])
        c = cmap(i / max(len(keys) - 1, 1))
        ax.plot(tr[:, 0], tr[:, 1], color=c, alpha=0.5, lw=1.1)
        ax.scatter(tr[-1, 0], tr[-1, 1], color=c, s=MARK_END, zorder=5,
                   edgecolors='black', linewidths=MARK_EDGE)
    ax.scatter([INIT[0]], [INIT[1]], marker='P', s=MARK_INIT, c='black', zorder=8)
    for _, (a, b) in FITS.items():
        ax.scatter([a], [b], marker='*', s=MARK_STAR, c='black', zorder=8)
    ax.legend(handles=marker_handles(), loc='best', fontsize=STYLE['legend'], framealpha=0.85,
              handlelength=1.4, columnspacing=0.8, borderpad=0.3, labelspacing=0.25)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1)),
                      ax=ax, fraction=0.05, pad=0.02)
    cb.set_label('relative depth', fontsize=STYLE['tick'])
    cb.ax.tick_params(labelsize=STYLE['tick'])
    panel_letter(ax, 'B')
    finish(fig, ax)
    for ext in ('pdf', 'png'):
        fig.savefig(HERE / f'fig_traj_sites{SUF}.{ext}', dpi=300)
    plt.close(fig)
    # ---------------- figure 3: the mixture inside the hull of the fits ------
    fits = np.array([FITS['GELU'], FITS['Swish'], FITS['Mish']])
    fig, ax = plt.subplots(figsize=STYLE['size'])
    hull = np.vstack([fits, fits[:1]])
    ax.fill(hull[:, 0], hull[:, 1], color='0.6', alpha=0.18, zorder=0)
    ax.plot(hull[:, 0], hull[:, 1], color='0.35', lw=0.9, ls='--', zorder=1)
    seen3 = []
    for label, arm, cell, tier, col in MIX:
        dirs = sorted((ROOT / 'controls6').glob(f'{arm}_{cell}_s1{tier}'))
        if not dirs:
            continue
        tr = np.vstack([mix_traj(dirs[0] / 'history.json')])
        ax.plot(tr[:, 0], tr[:, 1], color=col, alpha=0.6, lw=0.9, zorder=4)
        ax.scatter(tr[-1, 0], tr[-1, 1], color=col, s=MARK_END, zorder=6,
                   edgecolors='black', linewidths=MARK_EDGE)
        seen3.append((label, col))
    ax.scatter([INIT[0]], [INIT[1]], marker='P', s=MARK_INIT, c='black', zorder=7)
    for _, (a, b) in FITS.items():
        ax.scatter([a], [b], marker='*', s=MARK_STAR, c='black', zorder=7)
    handles = ([Line2D([], [], color=c, lw=1.6, label=l) for l, c in seen3]
               + [Line2D([], [], color='0.35', lw=1.2, ls='--', label='reachable set (hull)')]
               + marker_handles())
    ax.legend(handles=handles, loc='upper right', ncol=1, fontsize=STYLE['mix_legend'],
              framealpha=0.85, handlelength=1.2, columnspacing=0.6, borderpad=0.25,
              labelspacing=0.2)
    ax.set_xlim(0.11, 0.39)
    ax.set_ylim(-0.03, 0.21)
    panel_letter(ax, 'C')
    finish(fig, ax)
    for ext in ('pdf', 'png'):
        fig.savefig(HERE / f'fig_traj_mix{SUF}.{ext}', dpi=300)
    plt.close(fig)

    print(f'wrote fig_traj_single{SUF}, fig_traj_sites{SUF} and fig_traj_mix{SUF} (pdf+png); '
          f'{len(keys)} sites, {len(seen3)} mixture arms')


def run_set(style, suf):
    global SUF
    SUF = suf
    use_style(style)
    main()


if __name__ == '__main__':
    run_set(SINGLE_STYLE, '')

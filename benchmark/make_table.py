"""Turn a bench_efficiency.py results.json into tables.

    python make_table.py results.json                  # readable table (all columns)
    python make_table.py results.json --latex          # LaTeX rows for the paper
    python make_table.py results.json --patch ../paper/temp.tex
    python make_table.py results.json --plain

The readable table reports both training columns - the measured ms/step and the
projected seconds per epoch (ms/step x 97.6) - alongside the raw activation
timings.  The LaTeX rows only carry what the paper's table has: forward,
backward, seconds per epoch and the training ratio.

Each row is:  activation & forward (ms) & backward (ms) & s/epoch & vs GELU
plus a trailing comment-free set of numbers, matching tab:perf.  `vs GELU` is the
*training* ratio (seconds per epoch), which is the column the paper quotes;
the raw ratio is available in the printed summary and in the JSON.

Rows are wrapped in \\added{} so they render in blue like the rest of the revised
material; use --plain for the final, unmarked version.
"""
import argparse
import json
import re

# label used in the benchmark -> label used in the paper
PAPER = {
    'GELU': 'GELU',
    'Swish': 'Swish',
    'Mish': 'Mish',
    'GPS (2p)': 'GPS',
    'mixture (3p)': 'Mixture',
    'free mix (9p)': 'free mixture',
}


def collect(data):
    micro = {r['arm']: r for r in data.get('micro', [])}
    train = {r['arm']: r for r in data.get('training', [])}
    rows = []
    for arm in micro:                                   # benchmark order
        if arm not in train:
            continue
        rows.append((PAPER.get(arm, arm), micro[arm], train[arm]))
    return rows


# arms the paper's efficiency table lists, in its order
PAPER_ARMS = ['GELU', 'Swish', 'Mish', 'GPS', 'Mixture']


def fmt_sd(x, nd=3):
    """An uncertainty below the printed resolution must not read as 0.000."""
    return f'{x:.{nd}f}' if x >= 5 * 10 ** (-(nd + 1)) else f'<{10 ** -nd:.{nd}f}'


def param_txt(t):
    """Activation parameters of the row: 'total (per layer)'."""
    n, pl = t.get('act_params'), t.get('params_per_layer')
    if not n:
        return '0'
    return f'{n} ({pl}/layer)' if pl else f'{n}'


def pm(mean, sd, dp):
    """$mean \\pm sd$ at dp decimals for the mean, or a bare mean if no sd."""
    if sd is None:
        return f'${mean:.{dp}f}$'
    return f'${mean:.{dp}f} \\pm {fmt_sd(sd)}$'


def rows_text(data, added=False, all_arms=False):
    # one \added{} group per *cell*: a group may not span alignment tabs
    wrap = (lambda s: r'\added{' + s + '}') if added else (lambda s: s)
    out = []
    for label, m, t in collect(data):
        if not all_arms and label not in PAPER_ARMS:
            continue
        cells = [f'{label:13s}',
                 pm(m['fwd_ms'], m.get('fwd_sem'), 3),
                 pm(m['bwd_ms'], m.get('bwd_sem'), 3),
                 pm(t['s_epoch'], t.get('s_epoch_sd'), 2),
                 f'${t["vs_gelu"]:.3f}$',
                 f'${param_txt(t)}$']
        out.append('\t\t\t' + ' & '.join(wrap(c) for c in cells) + r' \\')
    return '\n'.join(out)


def markdown(data):
    """Readable table: raw activation plus both training columns."""
    env = data.get('env', {})
    cfg = data.get('config', {})
    out = []
    if env:
        out.append(f"GPU: {env.get('gpu', '?')}  |  torch {env.get('torch', '?')}  |  "
                   f"triton {env.get('triton', '?')}  |  {env.get('timestamp', '')}")
    if cfg:
        out.append(f"config: N={cfg.get('n')}, trials={cfg.get('trials')}, "
                   f"batch={cfg.get('batch')}, steps={cfg.get('steps')} x {cfg.get('reps')}")
    out += ['', '| activation | forward (ms) | backward (ms) | fwd+bwd (ms) | ms/step | '
                'time/epoch (s) | vs GELU (train) | act params |',
            '|---|---|---|---|---|---|---|---|',
            '| | ± s.e.m. | ± s.e.m. | | ± sd over repeats | ± sd over repeats | | |']
    for label, m, t in collect(data):
        ep = t.get('s_epoch_sd')
        ep_txt = (f'{t["s_epoch"]:.2f} ± {fmt_sd(ep)}' if ep is not None
                  else f'{t["s_epoch"]:.2f}')
        def raw(key, sd_key):
            sd = m.get(sd_key)
            return f'{m[key]:.3f}' + (f' ± {fmt_sd(sd)}' if sd is not None else '')
        out.append(f'| {label} | {raw("fwd_ms", "fwd_sem")} | {raw("bwd_ms", "bwd_sem")} | '
                   f'{m.get("fwd_bwd_ms", float("nan")):.3f} | '
                   f'{t["ms_step"]:.2f} ± {fmt_sd(t["ms_step_sd"])} | '
                   f'{ep_txt} | {t["vs_gelu"]:.3f} | {t.get("act_params", "")} |')
    return '\n'.join(out)


def patch(path, data, added=False, all_arms=False):
    src = open(path).read()
    m = re.search(r'(\\label\{tab:perf\}.*?\\midrule\n)(.*?)(\n\s*\\bottomrule)', src, re.S)
    if not m:
        raise SystemExit('tab:perf table not found in ' + path)
    open(path, 'w').write(src[:m.start(2)] + rows_text(data, added, all_arms) + src[m.end(2):])
    print(f'patched {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', help='results.json written by bench_efficiency.py')
    ap.add_argument('--latex', action='store_true', help='print LaTeX rows instead of a table')
    ap.add_argument('--patch', metavar='TEX', help='rewrite the tab:perf rows in this file')
    ap.add_argument('--added', action='store_true', help='wrap every cell in \\added{}')
    ap.add_argument('--all-arms', action='store_true',
                    help='include arms the paper table does not list (free mixture)')
    a = ap.parse_args()
    data = json.load(open(a.results))
    if a.patch:
        patch(a.patch, data, a.added, a.all_arms)
    elif a.latex:
        print(rows_text(data, a.added, a.all_arms))
    else:
        print(markdown(data))
    env = data.get('env', {})
    if env and not a.latex and not a.patch:
        import sys
        print(f'GPU: {env.get("gpu", "?")} | torch {env.get("torch", "?")} | '
              f'triton {env.get("triton", "?")} | {env.get("timestamp", "")}', file=sys.stderr)


if __name__ == '__main__':
    main()

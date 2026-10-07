"""beta-Swish baselines on the text and regression cells (tab:accs, rows 1-8).

    python run_baselines_nonvision.py                  # 8 cells x 5 seeds = 40 runs
    python run_baselines_nonvision.py --dry-run
    python run_baselines_nonvision.py --cells wikitext,wikitext_m50 --seeds 1,2,3

~1.6 GPU-h.  Artifacts land in <out>/beta_swish_wikitext_s1/, ..._s1_m50/,
..._s1_big/ etc., exactly like the main suite's naming so both trees aggregate
with the same tools.  Runs are resumable.

Epoch counts and corpus slices come from the cell matrix (8/4/2 epochs on the
12/50/100 MB WikiText tiers, 8/2 on OpenOrca, 50 on California housing); the
export reports California housing as -MSE so that higher is always better.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner import run_nonvision

# --------------------------- the test: what to run ---------------------------
CONFIG = dict(
    cells=['wikitext', 'wikitext_m50', 'wikitext_big',
           'openorca', 'openorca_big',
           'wikitext_glu', 'openorca_glu',
           'calhousing'],
    arms=['beta_swish'],           # trainable_acts.ARMS; add more via --arms
    seeds=[1, 2, 3, 4, 5],         # the paper's five seeds
    epochs=0,                      # 0 = per-cell counts from the cell matrix
    share='none',                  # one parameter set per activation site
    mult=1.0,                      # activation lr = 1.0 x weight lr
    out='controls_baselines',
)
# the training recipe and the cell definitions are in ../recipe.py

if __name__ == '__main__':
    run_nonvision(CONFIG, __doc__)

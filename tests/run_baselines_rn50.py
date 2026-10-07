"""beta-Swish baselines on ResNet-50 / CIFAR-100 (tab:accs, row 11).

    python run_baselines_rn50.py                       # 5 seeds = 5 runs
    python run_baselines_rn50.py --dry-run
    python run_baselines_rn50.py --arms beta_swish,prelu_layer

~2.5 GPU-h.  Artifacts land in <out>/beta_swish_rn50c100_s1/ ... and every run
is resumable.  The two ResNet-18 halves are run_baselines_rn18_c10.py and
run_baselines_rn18_c100.py.

Note the batch: the baseline suite trains this cell at batch 128 (the paper's
ResNet-50 setting) and names its artifacts without a suffix; the main suite's
batch-256 runs are plain 'rn50c100' and its batch-128 runs are 'rn50c100_b128'
(artifact suffix _b128).  The batch map below reproduces the baseline naming.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner import run_vision

# --------------------------- the test: what to run ---------------------------
CONFIG = dict(
    cells=['rn50c100'],            # recipe.CELLS: resnet50 / cifar100
    batch={'rn50c100': 128},       # the paper's ResNet-50 batch (matrix says 256)
    arms=['beta_swish'],           # trainable_acts.ARMS; add more via --arms
    seeds=[1, 2, 3, 4, 5],         # the paper's five seeds
    epochs=200,
    share='none',                  # one parameter set per activation site
    mult=1.0,                      # activation lr = 1.0 x weight lr
    out='controls_baselines',
)
# the training recipe and the cell definitions are in ../recipe.py

if __name__ == '__main__':
    run_vision(CONFIG, __doc__)

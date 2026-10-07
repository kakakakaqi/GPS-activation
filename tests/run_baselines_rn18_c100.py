"""beta-Swish baselines on ResNet-18 / CIFAR-100 (tab:accs, row 10).

    python run_baselines_rn18_c100.py                   # 5 seeds = 5 runs
    python run_baselines_rn18_c100.py --dry-run
    python run_baselines_rn18_c100.py --arms beta_swish,prelu_layer

~2.4 GPU-h.  Artifacts land in <out>/beta_swish_rn18c100_s1/ ... and every run
is resumable.  The CIFAR-10 half is run_baselines_rn18_c10.py, the ResNet-50
half is run_baselines_rn50.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner import run_vision

# --------------------------- the test: what to run ---------------------------
CONFIG = dict(
    cells=['rn18c100'],            # recipe.CELLS: resnet18 / cifar100
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

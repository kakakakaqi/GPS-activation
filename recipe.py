"""What every test of the paper shares: the training recipe and the benchmark
cells.  The per-test choices (which arms, which seeds, which batch, output dir)
are in the CONFIG block at the top of each script in tests/.

Two layers live here and nowhere else:

  RECIPE / ACT_PARAMS / AUG_*  - *how* we train (the paper's training scheme):
    AdamW at lr 1e-3 with weight decay 0.1, cosine schedule over 5% warmup,
    grad clip 1.0, full reproducibility.  The activation's own parameters get
    no weight decay and a lr of act_lr_mult x the weight lr, cosine-annealed to
    0 at the end of the run.  The vision cells add RandAugment(2, 9), MixUp
    (alpha 0.2) and label smoothing 0.1; the text and regression cells train
    unaugmented.

  CELLS - *what* we train on: one entry per benchmark cell of tab:accs, with
    its architecture, dataset, batch size, epoch count, corpus slice and the
    suffix its artifact directories carry.  Corpus tiers are separate cells
    ('wikitext_m50', 'wikitext_big', ...) so a run needs no environment knobs.

Artifact directories are named <arm>_<stem>_s<seed><suffix>; 'stem' defaults to
the cell key and 'suffix' to the empty string.

Note on ResNet-50: the main suite ran it at batch 256 ('rn50c100') and again at
batch 128 ('rn50c100_b128', artifact suffix '_b128'); the baseline suite ran
its batch-128 runs under the plain name.  tab:accs reports the batch-128
numbers, i.e. 'rn50c100_b128' for the main suite and ('rn50c100', batch=128)
for the baselines - see the CONFIG blocks in tests/.
"""

# --------------------------------------------------------------- the recipe
# 'share' is the fallback parameter-sharing mode; arms that need per-site
# parameters override it (see the arm table in tests/main_suite.py).
RECIPE = dict(optimizer='adamw', lr=1e-3, weight_decay=0.1, warmup_frac=0.05,
              lr_schedule='cosine', grad_clip=1.0, full_reproducibility=True,
              mixup_alpha=0.0, label_smoothing=0.0, verbose=True,
              data_dir='./data', share='global', freeze_schedule='')

# the activation's own parameters: never decayed (decaying them biases the
# shape search), lr = act_lr_mult x weight lr, cosine to 0 at the end of the run
ACT_PARAMS = dict(act_weight_decay=0.0, act_decay_until=1.0)

# ------------------------------------------------------------- augmentation
AUG_VISION = dict(augmentation='randaugment',
                  augmentation_kwargs={'num_ops': 2, 'magnitude': 9},
                  mixup_alpha=0.2, label_smoothing=0.1)
AUG_NONE = dict(augmentation='none')

# ------------------------------------------------------------- the cells
CELLS = {
    # ---- vision (tab:accs rows 9-11) ----
    'rn18':           dict(arch='resnet18', dataset='cifar10',  batch=256,
                           epochs=200, aug='vision'),
    'rn18c100':       dict(arch='resnet18', dataset='cifar100', batch=256,
                           epochs=200, aug='vision'),
    'rn50c100':       dict(arch='resnet50', dataset='cifar100', batch=256,
                           epochs=200, aug='vision'),
    'rn50c100_b128':  dict(stem='rn50c100', suffix='_b128',
                           arch='resnet50', dataset='cifar100', batch=128,
                           epochs=200, aug='vision'),
    # ---- text (tab:accs rows 1-7), corpus tiers as their own cells ----
    'wikitext':       dict(arch='charlm', dataset='wikitext', batch=64,
                           epochs=8, aug='none', corpus=12_000_000),
    'wikitext_m50':   dict(stem='wikitext', suffix='_m50',
                           arch='charlm', dataset='wikitext', batch=64,
                           epochs=4, aug='none', corpus=50_000_000),
    'wikitext_big':   dict(stem='wikitext', suffix='_big',
                           arch='charlm', dataset='wikitext', batch=64,
                           epochs=2, aug='none', corpus=100_000_000),
    'openorca':       dict(arch='charlm', dataset='openorca', batch=64,
                           epochs=8, aug='none', corpus=12_000_000),
    'openorca_big':   dict(stem='openorca', suffix='_big',
                           arch='charlm', dataset='openorca', batch=64,
                           epochs=2, aug='none', corpus=100_000_000),
    'wikitext_glu':   dict(arch='charlm_glu', dataset='wikitext', batch=64,
                           epochs=8, aug='none', corpus=12_000_000),
    'openorca_glu':   dict(arch='charlm_glu', dataset='openorca', batch=64,
                           epochs=8, aug='none', corpus=12_000_000),
    # ---- regression (tab:accs row 8) ----
    'calhousing':     dict(arch='regmlp', dataset='calhousing', batch=256,
                           epochs=50, aug='none', eval_metric='mse'),
    # ---- the local 2 MB tech-docs corpus (in the artifacts, no tab:accs row) ----
    'charlm':         dict(arch='charlm', dataset='charlm', batch=64,
                           epochs=8, aug='none'),
}

# GPS — Gaussian Probability Softplus

A trainable activation function that attempts to match the performance of the best among GELU, Swish or
Mish for any task, so that the choice between them stops being a hyperparameter (see results for accuracy and efficiency).

```
f(x) = softplus(x) − ln2 · exp(−a x² − b x)                      2-parameter form
f(x) = softplus(x) − ln2 · Σᵢ wᵢ exp(−aᵢ x² − bᵢ x)               mixture form
```

Paper: *GPS: A Fast Trainable Activation Function* (Zhang, 2026). This repository
is the code that produced every number in it. 

*The paper has not been published yet*

---

## Install

```
torch>=2.4   torchvision>=0.19   triton>=3.0   numpy>=1.24   pyarrow>=15
```

`lib/gps.py` runs on plain PyTorch, and switches to the fused Triton kernel
automatically when Triton is installed and the tensors are CUDA `float32`. 

---

## Standard implementation

```python
import torch
import gps

model = ...                                        # any network with GELU/SiLU/Mish
gps.swap_activations(model)                        # 2-parameter form by default
gps.share_parameters(gps.find(model), 'global')    # one (a, b) for the whole net

model = model.cuda()                               # activation params stay float32
gps.warmup(model)                                  # compile the fused Triton kernel

weight_lr = 1e-3
opt = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'params' not in n],
     'lr': weight_lr},
    {'params': gps.find(model)[0].params,
     'lr': weight_lr * gps.PROBE_MULTIPLIER},      # 1.0: epoch 1 is the probe
])

for epoch in range(200):
    if epoch == 1:                                 # one epoch has been trained
        gps.calibrate(model, opt.param_groups[1], epochs=200)
    train_one_epoch()
```

`swap_activations` follows the model's device, so it works equally well before
or after `.cuda()`; `warmup` runs a throwaway forward and backward so that the
kernel's compilation and autotuning do not land on the first real step. 

`share_parameters(..., 'global')` makes GPS share parameters. Not used in the paper.

---

## Mixture form implementation (convolutional networks)

```python
gps.swap_activations(model, form='mixture')       # 3 logits instead of 2 params

# ... train, with λ fixed at 0.5; no calibration

gps.fold(model)       # every site -> its 2-parameter equivalent, in place
# ⚠ batch-norm statistics still belong to the mixture — recalibrate them
```

`gps.fold(model)` solves `(a, b) = argmin MSE(f_mixture, f_2param)` on `[-8, 8]`
for each site — the paper's map φ(w) — and converts the modules in place, so the
same model object now runs the two-parameter form with two trainable numbers per
site instead of three. It returns `(a, b, mse)` per site if you want to inspect
the result, and warns about the batch norm: the fold changes the curve but not
the statistics, so run a forward pass over the training data in `eval()` mode
afterwards, as the paper does.

The mixture's λ is **not calibrated** — it is fixed at 0.5, and the sweep that
justified 0.5 (`λ ∈ {0.125, 0.25, 0.5}`) is exactly the kind of search the probe
is meant to eliminate. The probe has no meaning there yet: it measures a
displacement in the two-parameter curve space, and the mixture's trainable
objects are three logits initialized at `w* = (0.361, 0.307, 0.332)`. Making the
probe work for the mixture requires defining what it measures; nobody has.

---

## Calibrating the activation learning rate

The activation's parameters are trained at `weight_lr × λ` there is no single λ that works well. There are two ways to calibrate.

### A — without restart (continue)

The cheaper form. Train the first epoch at λ = 1 (the probe), then set λ and
keep going:

```python
for epoch in range(200):
    if epoch == 1:                                 # one epoch has been trained
        gps.calibrate(model, opt.param_groups[1], epochs=200)
    train_one_epoch()
```

### B — with restart (what the paper reports)

Run one probe epoch at λ = 1, throw it away, and train the whole run again at the
calibrated λ. Costs one epoch, and is what the paper uses. 

```python
import copy

before = copy.deepcopy(model.state_dict())          # the initialisation

def probe_epoch():
    ...                                             # one epoch at act lr == weight lr

lam = gps.calibrate_lr_multiplier(model, probe_epoch, epochs=200, remaining=200)
model.load_state_dict(before)                       # discard the probe epoch
gps.set_multiplier(model, lam)                      # ⇒ weight_lr * lam from epoch 0
```

---

## Results

| Benchmark | GELU | Swish | Mish | **GPS** | Mixed Wave GPS | β-Swish |
|---|---|---|---|---|---|---|
| WikiText-103, 12 MB | 61.04±0.24 | 60.47±0.30 | 60.53±0.30 | **61.11±0.29** | 60.76±0.30 | 60.78±0.26 |
| WikiText-103, 50 MB | 62.33±0.22 | 61.98±0.27 | 61.99±0.31 | **62.45±0.18** | 62.16±0.28 | 62.27±0.27 |
| WikiText-103, 100 MB | 62.73±0.23 | 62.17±0.15 | 62.23±0.23 | **62.80±0.16** | 62.57±0.25 | 62.56±0.14 |
| OpenOrca, 12 MB | **60.54±0.23** | 59.76±0.15 | 59.82±0.16 | 60.50±0.20 | 60.19±0.17 | 60.25±0.25 |
| OpenOrca, 100 MB | **62.95±0.38** | 61.89±0.38 | 62.06±0.43 | 62.93±0.42 | 62.43±0.49 | 62.43±0.43 |
| WikiText-103 GLU, 12 MB | 62.34±0.22 | 62.27±0.13 | 62.37±0.14 | **62.41±0.15** | 62.33±0.20 | 62.32±0.22 |
| OpenOrca GLU, 12 MB | 61.96±0.20 | 61.84±0.19 | 61.94±0.20 | **62.07±0.18** | 62.03±0.15 | 61.94±0.27 |
| California housing (MSE, lower better) | **0.27±0.00** | 0.28±0.00 | 0.28±0.00 | **0.27±0.00** | 0.28±0.00 | 0.28±0.00 |
| CIFAR-10 / ResNet-18 | 96.28±0.11 | 96.14±0.07 | 96.12±0.10 | 96.32±0.14 | **96.40±0.09** | 96.28±0.09 |
| CIFAR-100 / ResNet-18 | 79.39±0.31 | 79.58±0.24 | 79.44±0.11 | 78.20±0.23 | **79.66±0.14** | 78.86±0.25 |
| CIFAR-100 / ResNet-50 | 77.17±0.35 | 76.95±0.15 | **77.33±0.20** | 76.36±0.31 | **77.33±0.25** | 76.91±0.08 |

### Efficiency

Forward and backward time per activation on a vector of 6.7e7 float32 elements,
averaged over 100 trials on an RTX 5090; time per epoch from ResNet-18 on
CIFAR-10 at batch 512, which is the number that actually matters.

| Activation | Forward (ms) | Backward (ms) | Time/epoch (s) | vs. GELU |
|---|---|---|---|---|
| GELU | 0.352 ± 0.001 | 0.510 ± 0.001 | 4.64 ± 0.001 | 1.000 |
| Swish | 0.352 ± 0.001 | 0.511 ± 0.001 | 4.78 ± 0.001 | 1.030 |
| Mish | 0.352 ± 0.001 | 0.511 ± 0.001 | 4.78 ± 0.002 | 1.031 |
| GPS | 0.352 ± 0.001 | 0.519 ± 0.001 | 4.66 ± 0.001 | 1.004 |
| Mixed Wave GPS | 0.354 ± 0.001 | 0.535 ± 0.001 | 4.73 ± 0.001 | 1.019 |

---

## Repository layout

```
recipe.py            the training recipe and the benchmark cell matrix
lib/gps.py           the activation, both forms, fold, and the λ calibration  ← start here
lib/gps_lab.py       experiment runtime: trainer, registry, fused Triton kernels
lib/gps_choice.py    mixture-wave module (softmax / Gumbel weights)
lib/trainable_acts.py  trainable-activation baselines: beta_swish, beta_gelu,
                       eswish, prelu_layer, pau
lib/models.py        datasets and benchmark architectures
lib/runner.py        job expansion + artifact export
tests/               one entry script per benchmark family, hyperparameters inline
derive/              minimax initialization of (a, b) and the approximation fits
figures/             activation fits and parameter-trajectory figures
benchmark/           activation microbenchmark and the efficiency table
```

---

## Reproducing the paper

Every run is resumable: a job whose `summary.json` exists is skipped, so
interrupted grids restart where they stopped. `--dry-run` lists the jobs.

```bash
# main suite: statics, GPS, mixture and the calibrated arm, 5 seeds
python tests/run_main.py --cells rn18 --seeds 1,2,3,4,5 \
    --arms static_gelu,static_silu,static_mish,naive2p_mauto,mix_persite
python tests/run_main.py --cells wikitext,wikitext_m50,wikitext_big,openorca,openorca_big \
    --arms static_gelu,static_silu,static_mish,naive2p_mauto,mix_persite \
    --seeds 1,2,3,4,5

# trainable-activation baselines (β-Swish)
python tests/run_baselines_rn18_c10.py      # ~2.3 GPU-h
python tests/run_baselines_rn18_c100.py
python tests/run_baselines_rn50.py
python tests/run_baselines_nonvision.py     # ~1.6 GPU-h

# activation microbenchmark + the efficiency table
cd benchmark && ./run.sh          # or: bash run.sh
```

The calibrated arm replays the λ values in `PROBE_MULTIPLIER` (in
`tests/run_main.py`) so that a rerun reproduces the paper's numbers; `lib/gps.py`
implements the live calibration for new work.

### Difference from the paper's code

The `gps_lab.py` here has one line that the version used for the paper did not.
It fixes a bug.

The backward Triton kernel adds the gradient for `(a, b)` into a buffer with
atomics. Triton tunes a kernel by running it several times to see which
configuration is fastest, and those runs used the real buffer, so their
contributions piled up inside it. The real launch then added the true gradient on
top.

What it affects is narrow. Only the first backward pass for a given input size is
wrong; after that the choice is cached and nothing is benchmarked. And only the
gradient for `(a, b)` is wrong, because the gradient for `x` is overwritten
rather than added, so it comes out right every time. Even that gradient is mostly
harmless: Adam divides each coordinate by its running second moment, so a
constant factor on a gradient largely cancels in the update.

With 5000 standard normal inputs the reference gradient is 1955.269. The first
kernel backward gave 2,684,616, about 1373x too large. The second call onward
gave 1955.269. The factor is not fixed either: it is however many times the tuner
ran the kernel, so it depends on the machine. Across input sizes from 1e3 to 1e5
it came out between 1346x and 1365x.

The fix is `restore_value=['grad_params_ptr']` on the autotune decorator, which
saves the buffer before tuning and puts it back afterwards. The mixture kernel
didn't need it, since its backward doesn't use atomics.

What it causes in the paper is at most one oversized update per input size, at
the very start of a run: two updates out of roughly 39,000 for ResNet-18 on
CIFAR-10. The one path left for it to matter is the gradient clip: the affected
step is clipped to norm 1, which makes it no larger than an ordinary Adam step of
`lr_a = 1.25e-5`, against 2.8e-3 of travel in that epoch. At the precision the
paper reports, nothing moves. Code that uses SGD, or that reads gradient
magnitudes, would care, which is why this version keeps the fix. If you want the
exact code path from the paper, delete that one argument.

---

## Citation

```bibtex
@article{zhang_gps,
  title  = {GPS: A Fast Trainable Activation Function},
  author = {Zhang, Jerry},
  year   = {2026}
}
```

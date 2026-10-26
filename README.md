# OOD Detection and Selective Prediction

Code and reproducible milestone artifacts for our project on OOD detection and selective prediction. The current study uses two separate ID tasks (CIFAR-10 and CIFAR-100), ResNet-18 standard training and energy-aware fine-tuning, with SVHN and DTD as held-out OOD test sets.

## Repository contents

- `src/`: dataset loading/preparation, model, training, OOD scoring, selective prediction metrics, evaluation, aggregation, and smoke-check scripts.
- `data/tinyimagenet_exclude_*.txt`: manually curated Tiny ImageNet class exclusion lists for each CIFAR ID task. Dataset binaries are not included.
- `runs/results_summary.csv`: measured summary metrics for the current milestone.
- `runs/pre2_milestone_report.md`: detailed project progress and experiment report.
- The figures referenced by the report remain in AutoDL's `runs/figures/` directory and are not included in this repository.

## Current results

The strongest positive result is task-dependent: CIFAR-100 energy-aware fine-tuning with MSP raises SVHN AUROC from 0.609 to 0.787 while test accuracy changes from 0.721 to 0.692. On CIFAR-10, fine-tuning reduces OOD AUROC. Energy-aware training is therefore not a universal improvement; the accuracy, calibration, and OOD detection trade-offs are reported in the milestone report.

## Setup (AutoDL/Linux)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install a PyTorch build compatible with the AutoDL CUDA image if needed. Check the environment with `python -c "import torch; print(torch.cuda.is_available())"`.

## Data protocol

- CIFAR-10 and CIFAR-100 are independent ID tasks. Validation is drawn from the official training partition; official test data are held out.
- Tiny ImageNet training images belong under `data/tiny-imagenet-200/train/<wnid>/...`. Use the matching exclusion list in `data/` for each task.
- SVHN and DTD are used only as final OOD test sets, not for threshold selection or tuning.
- Data preparation details and experiment outputs are summarized in `runs/pre2_milestone_report.md`.

To download the five core datasets and create a portable ZIP on AutoDL:

```bash
python -m src.download_datasets --root data --bundle datasets_bundle.zip
```

## Reproduction

Train standard baselines:

```bash
python -m src.train --id-dataset cifar10 --output runs/cifar10-baseline --epochs 20
python -m src.train --id-dataset cifar100 --output runs/cifar100-baseline --epochs 20
```

Train energy-aware models after the corresponding baseline is ready:

```bash
python -m src.train --id-dataset cifar10 --energy-aware --aux-root data/tiny-imagenet-200/train --output runs/cifar10-energy --epochs 20
python -m src.train --id-dataset cifar100 --energy-aware --aux-root data/tiny-imagenet-200/train --output runs/cifar100-energy --epochs 20
```

Evaluate MSP or energy scores with a validation-selected threshold (default target coverage 95%). Higher confidence is normalized to mean “more ID-like”. See `python -m src.evaluate --help` for the full options and `runs/results_summary.csv` for current results.

Dataset binaries, checkpoints, and large per-run score arrays are excluded from this repository. The report distinguishes completed measurements from planned follow-up experiments.


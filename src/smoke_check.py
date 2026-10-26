"""Minimal smoke checks for the Pre2 pipeline (run after data preparation).

Verifies, without any training:
1. CIFAR-10/100 loaders: shapes, split sizes (45k/5k/10k), split reproducibility
   under a fixed seed, and that train/val indices do not overlap (no leakage).
2. CIFAR-style ResNet-18 forward pass and checkpoint save/load round-trip.
3. MSP / energy score direction: larger means more ID-like.
4. Metric implementations on synthetic scores with known answers.

Run: python -m src.smoke_check
"""
import json
import tempfile

import numpy as np
import torch

from src.data import id_loaders, ood_loaders, tinyimagenet_outliers
from src.metrics import detection_metrics, ece, risk_coverage
from src.model import make_model
from src.scores import id_confidence

results = {}


def check(name, ok, detail=""):
    results[name] = {"ok": bool(ok), "detail": str(detail)}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def check_loaders():
    for name, ncls in [("cifar10", 10), ("cifar100", 100)]:
        tr, va, te = id_loaders(name, "data", batch_size=256, seed=42, workers=2)
        check(f"{name}: loader sizes", len(tr.dataset) == 45000 and len(va.dataset) == 5000 and len(te.dataset) == 10000,
              f"train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}")
        x, y = next(iter(tr))
        check(f"{name}: train batch", tuple(x.shape) == (256, 3, 32, 32) and y.max() < ncls,
              f"shape={tuple(x.shape)} label_max={int(y.max())}")
        xv, _ = next(iter(va))
        check(f"{name}: val batch (eval tf, no aug)", tuple(xv.shape) == (256, 3, 32, 32), f"shape={tuple(xv.shape)}")
        # reproducibility of the split
        _, va2, _ = id_loaders(name, "data", batch_size=256, seed=42, workers=0)
        va_ds, va_ds2 = va.dataset, va2.dataset
        same = len(va_ds) == len(va_ds2) and all(
            np.array_equal(va_ds.indices, va_ds2.indices) if hasattr(va_ds, "indices") else False for _ in [0])
        check(f"{name}: val split reproducible (seed=42)", same, f"indices match: {same}")
        # no overlap between train and val
        tr_idx = set(tr.dataset.indices)
        va_idx = set(va.dataset.indices)
        check(f"{name}: train/val disjoint", not (tr_idx & va_idx), f"overlap={len(tr_idx & va_idx)}")


def check_model_and_scores():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ncls in [10, 100]:
        m = make_model(num_classes=ncls).to(device)
        x = torch.randn(8, 3, 32, 32, device=device)
        out = m(x)
        check(f"resnet18 forward ({ncls} cls)", tuple(out.shape) == (8, ncls), f"out={tuple(out.shape)}")
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"model": m.state_dict(), "id_dataset": "cifar10"}, f.name)
            m2 = make_model(num_classes=ncls).to(device)
            m2.load_state_dict(torch.load(f.name, map_location=device, weights_only=False)["model"])
            same = torch.allclose(m(x), m2(x), atol=1e-6)
            check(f"checkpoint save/load round-trip ({ncls} cls)", same, f"logits equal: {same}")
    # score direction
    logits_low = torch.tensor([[8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    logits_high = torch.tensor([[12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    msp_l, msp_h = id_confidence(logits_low, "msp"), id_confidence(logits_high, "msp")
    en_l, en_h = id_confidence(logits_low, "energy"), id_confidence(logits_high, "energy")
    check("msp direction (higher=more ID)", msp_h > msp_l and 0 < msp_l <= 1, f"{msp_l.item():.3f} < {msp_h.item():.3f}")
    check("energy direction (higher=more ID)", en_h > en_l, f"{en_l.item():.3f} < {en_h.item():.3f}")
    check("energy = T*logsumexp(logits/T)", abs(en_l.item() - torch.logsumexp(logits_low, 1).item()) < 1e-5,
          f"energy={en_l.item():.4f}")


def check_metrics():
    # well-separated scores: perfect detection
    ids = np.linspace(0.9, 1.0, 500)
    oods = np.linspace(0.0, 0.1, 500)
    m = detection_metrics(ids, oods)
    check("detection_metrics perfect separation", m["auroc_id_positive"] == 1.0 and m["fpr95_ood"] == 0.0, str(m))
    # identical distributions: chance level
    rng = np.random.default_rng(0)
    ids = rng.random(500)
    oods = rng.random(500)
    m = detection_metrics(ids, oods)
    check("detection_metrics chance level", 0.45 < m["auroc_id_positive"] < 0.55, f"auroc={m['auroc_id_positive']:.3f}")
    # ece: perfectly calibrated uniform case
    probs = np.tile(np.eye(10), (100, 1))
    labels = np.arange(1000) % 10
    check("ece one-hot confident+correct = 0", ece(probs, labels) < 1e-9, f"ece={ece(probs, labels)}")
    # risk coverage: monotone risk decreasing for perfect model
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([1, 1, 1, 0, 1], dtype=bool)
    rows = risk_coverage(conf, correct, points=5)
    check("risk_coverage ordering", rows[0][1] <= rows[-1][1] and abs(rows[0][0] - 0.2) < 1e-9, str(rows[:2]))


def check_ood_data():
    try:
        oods = ood_loaders("data", batch_size=256, workers=2)
        for name, dl in oods.items():
            n = len(dl.dataset)
            x, _ = next(iter(dl))
            check(f"ood {name} loader", n > 0 and tuple(x.shape)[1:] == (3, 32, 32), f"n={n} shape={tuple(x.shape)}")
    except Exception as e:
        check("ood loaders", False, f"{type(e).__name__}: {e}")
    try:
        dl = tinyimagenet_outliers("data/tiny-imagenet-200/train", batch_size=256, workers=2,
                                   exclude_file="data/tinyimagenet_exclude.txt")
        n_all = len(dl.dataset)
        check("tiny-imagenet outliers loader", n_all >= 0, f"kept={n_all} (exclusion list may be empty)")
    except Exception as e:
        check("tiny-imagenet loader", False, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    check_loaders()
    check_model_and_scores()
    check_metrics()
    check_ood_data()
    n_fail = sum(1 for r in results.values() if not r["ok"])
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    with open("runs/smoke_check.json", "w") as f:
        json.dump(results, f, indent=2)


"""Aggregate Pre2 experiment results into comparison tables and figures.

Runs layout:
- Standard baselines: runs/{task}-baseline
- Energy-aware from-scratch (pilot): runs/{task}-energy (+ w002 variants)
- Energy-aware fine-tuned: runs/{task}-energy-ft{K}, selected per task by ID
  validation accuracy only: cifar10 -> ft2 (w=0.02, 5 ep), cifar100 -> ft4 (w=0.005, 5 ep)

Outputs: results_summary.csv, comparison_tables.md, tuning CSVs,
figures/{score_distributions,risk_coverage,training_curves,finetune_tradeoff}.png

Run: python -m src.aggregate_results
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path("runs")
FIGS = RUNS / "figures"
TASKS = ["cifar10", "cifar100"]
MODELS = {"baseline": "Standard", "energy": "Energy-aware(scratch)"}
METHODS = {"msp": "MSP", "energy": "Energy"}
FT_SELECTED = {"cifar10": "cifar10-energy-ft2", "cifar100": "cifar100-energy-ft4"}
FT_W = {"cifar10": 0.02, "cifar100": 0.005}
FT_VARIANTS = {
    "cifar10": [("cifar10-energy-ft", 0.05, 10), ("cifar10-energy-ft2", 0.02, 5)],
    "cifar100": [("cifar100-energy-ft", 0.05, 10), ("cifar100-energy-ft2", 0.02, 5),
                 ("cifar100-energy-ft3", 0.01, 5), ("cifar100-energy-ft4", 0.005, 5)],
}


def metrics(run):
    return json.loads((RUNS / run / "metrics.json").read_text())


def row_from_metrics(task, model, method, run):
    m = metrics(run)
    return {
        "task": task, "model": model, "method": METHODS[method],
        "id_test_acc": m["id_test_accuracy"], "ece": m["id_test_ece"],
        "sel_cov@95": m["selective"]["accepted_coverage"],
        "sel_err@95": m["selective"]["accepted_error"],
        "svhn_auroc": m["ood"]["svhn"]["auroc_id_positive"],
        "svhn_aupr": m["ood"]["svhn"]["aupr_id_positive"],
        "svhn_fpr95": m["ood"]["svhn"]["fpr95_ood"],
        "dtd_auroc": m["ood"]["dtd"]["auroc_id_positive"],
        "dtd_aupr": m["ood"]["dtd"]["aupr_id_positive"],
        "dtd_fpr95": m["ood"]["dtd"]["fpr95_ood"],
    }


def load_all():
    rows = []
    for task in TASKS:
        for model in MODELS:
            for method in METHODS:
                rows.append(row_from_metrics(task, MODELS[model], method, f"{task}-{model}-{method}"))
        ft = FT_SELECTED[task]
        for method in METHODS:
            r = row_from_metrics(task, f"Energy-aware(FT,w={FT_W[task]})", method, f"{ft}-{method}")
            rows.append(r)
    return pd.DataFrame(rows)


def fmt_table(df, models_order, title, lines):
    lines.append(f"## {title}\n")
    lines.append("| Setting | ID acc | ECE | Sel.Err@95cov | SVHN AUROC | SVHN FPR95 | DTD AUROC | DTD FPR95 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for model in models_order:
        for method in METHODS.values():
            r = df[(df.model == model) & (df.method == method)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(
                f"| {model}+{method} | {r.id_test_acc:.4f} | {r.ece:.4f} | {r['sel_err@95']:.4f} "
                f"| {r.svhn_auroc:.4f} | {r.svhn_fpr95:.4f} | {r.dtd_auroc:.4f} | {r.dtd_fpr95:.4f} |")
    lines.append("")


def tables(df: pd.DataFrame):
    lines = ["# Pre2 comparison tables (thresholds from ID validation, 95% coverage)\n"]
    main_models = ["Standard", f"Energy-aware(FT,w={FT_W['cifar10']})", f"Energy-aware(FT,w={FT_W['cifar100']})"]
    for task in TASKS:
        sub = df[df.task == task]
        mm = [m for m in main_models if not sub[sub.model == m].empty]
        fmt_table(sub, mm, f"{task.upper()} — main 2x2 (selected energy-aware = fine-tuned)", lines)
    scratch_models = ["Standard", "Energy-aware(scratch)"]
    for task in TASKS:
        sub = df[df.task == task]
        fmt_table(sub, scratch_models, f"{task.upper()} — pilot: energy-aware trained from scratch (w=0.1, 20 ep)", lines)
    (RUNS / "comparison_tables.md").write_text("\n".join(lines))
    print("wrote runs/comparison_tables.md")


def fig_score_dists():
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    for i, task in enumerate(TASKS):
        panels = [(f"{task}-baseline-msp", "MSP"), (f"{task}-baseline-energy", "Energy"),
                  (f"{FT_SELECTED[task]}-msp", "MSP"), (f"{FT_SELECTED[task]}-energy", "Energy")]
        titles = [f"{task} Standard", "", f"{task} Energy-aware(FT)", ""]
        for k, (run, method) in enumerate(panels):
            ax = axes[i, k]
            is_energy = method.lower() == "energy"
            z = np.load(RUNS / run / "scores.npz")
            if is_energy:
                lo = min(z["id_test"].min(), z["ood_svhn"].min(), z["ood_dtd"].min())
                hi = max(z["id_test"].max(), z["ood_svhn"].max(), z["ood_dtd"].max())
                bins = np.linspace(lo, hi, 70)
            else:
                bins = np.linspace(0, 1, 70)
            ax.hist(z["id_test"], bins=bins, alpha=0.6, density=True, label="ID test")
            ax.hist(z["ood_svhn"], bins=bins, alpha=0.6, density=True, label="SVHN")
            ax.hist(z["ood_dtd"], bins=bins, alpha=0.6, density=True, label="DTD")
            ax.set_title(f"{titles[k]} + {method}", fontsize=10)
            if k == 0:
                ax.set_ylabel("density")
            if i == 1:
                ax.set_xlabel("energy score" if is_energy else "MSP")
            if i == 0 and k == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Confidence score distributions (higher = more ID-like)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "score_distributions.png", dpi=140)
    plt.close(fig)
    print("wrote figures/score_distributions.png")


def fig_risk_coverage():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, task in enumerate(TASKS):
        curves = [(f"{task}-baseline-msp", "Standard+MSP"), (f"{task}-baseline-energy", "Standard+Energy"),
                  (f"{FT_SELECTED[task]}-msp", "Energy-aware(FT)+MSP")]
        for run, label in curves:
            rc = pd.read_csv(RUNS / run / "risk_coverage.csv")
            axes[i].plot(rc.coverage, rc.risk, label=label, linewidth=1.6)
        axes[i].set_title(task.upper())
        axes[i].set_xlabel("coverage")
        axes[i].set_ylabel("selective risk")
        axes[i].grid(alpha=0.3)
        axes[i].legend(fontsize=8)
    fig.suptitle("Risk-coverage curves on ID official test", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "risk_coverage.png", dpi=140)
    plt.close(fig)
    print("wrote figures/risk_coverage.png")


def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, task in enumerate(TASKS):
        for run, label in [(f"{task}-baseline", "Standard"),
                           (f"{task}-energy", "Energy-aware(scratch, w=0.1)")]:
            h = json.loads((RUNS / run / "history.json").read_text())
            eps = [r["epoch"] for r in h["history"]]
            axes[i].plot(eps, [r["val_acc"] for r in h["history"]],
                         label=f"{label} (best {h['best_val_acc']:.3f})", linewidth=1.6)
        h = json.loads((RUNS / FT_SELECTED[task] / "history.json").read_text())
        eps = [r["epoch"] for r in h["history"]]
        base_ep = 20
        axes[i].plot([base_ep + e for e in eps], [r["val_acc"] for r in h["history"]],
                     marker="o", markersize=3, label=f"Energy-aware(FT) (best {h['best_val_acc']:.3f})", linewidth=1.6)
        axes[i].axvline(base_ep + 0.5, color="gray", linestyle=":", linewidth=1)
        axes[i].set_title(task.upper())
        axes[i].set_xlabel("epoch (FT starts after baseline epoch 20)")
        axes[i].set_ylabel("validation accuracy")
        axes[i].grid(alpha=0.3)
        axes[i].legend(fontsize=8)
    fig.suptitle("Training and fine-tuning curves (seed 42)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "training_curves.png", dpi=140)
    plt.close(fig)
    print("wrote figures/training_curves.png")


def fig_finetune_tradeoff():
    """Energy weight vs (val acc, SVHN/DTD AUROC@MSP) per task, with baseline reference lines."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, task in enumerate(TASKS):
        base_h = json.loads((RUNS / f"{task}-baseline" / "history.json").read_text())
        base_m = metrics(f"{task}-baseline-msp")
        ws, accs, svhn, dtd = [], [], [], []
        for run, w, _ep in FT_VARIANTS[task]:
            h = json.loads((RUNS / run / "history.json").read_text())
            m = metrics(f"{run}-msp")
            ws.append(w); accs.append(h["best_val_acc"])
            svhn.append(m["ood"]["svhn"]["auroc_id_positive"]); dtd.append(m["ood"]["dtd"]["auroc_id_positive"])
        ax = axes[i]
        ax.plot(ws, accs, marker="o", color="tab:blue", label="val accuracy")
        ax.axhline(base_h["best_val_acc"], color="tab:blue", linestyle=":", linewidth=1, label="Standard baseline acc")
        ax.set_xscale("log"); ax.set_xlabel("energy_weight (fine-tune)")
        ax.set_ylabel("val accuracy", color="tab:blue"); ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(ws, svhn, marker="s", color="tab:red", label="SVHN AUROC (MSP)")
        ax2.plot(ws, dtd, marker="^", color="tab:orange", label="DTD AUROC (MSP)")
        ax2.axhline(base_m["ood"]["svhn"]["auroc_id_positive"], color="tab:red", linestyle=":", linewidth=1)
        ax2.axhline(base_m["ood"]["dtd"]["auroc_id_positive"], color="tab:orange", linestyle=":", linewidth=1)
        ax2.set_ylabel("OOD AUROC", color="tab:red"); ax2.tick_params(axis="y", labelcolor="tab:red")
        for w, s_ in zip(ws, svhn):
            ax2.annotate(f"{s_:.2f}", (w, s_), textcoords="offset points", xytext=(0, 7), fontsize=7, color="tab:red")
        ax.set_title(task.upper())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Energy-aware fine-tuning trade-off (dotted = Standard baseline)", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGS / "finetune_tradeoff.png", dpi=140)
    plt.close(fig)
    print("wrote figures/finetune_tradeoff.png")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    df = load_all()
    df.to_csv(RUNS / "results_summary.csv", index=False)
    print(df.to_string(index=False))
    tables(df)
    fig_score_dists()
    fig_risk_coverage()
    fig_training_curves()
    fig_finetune_tradeoff()


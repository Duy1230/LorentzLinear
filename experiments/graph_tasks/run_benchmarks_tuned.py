"""Tier 2 Benchmarks -- tuned run with higher d_hyp and longer training."""
from __future__ import annotations

import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
import numpy as np

from data.loader import load_dataset
from models.hyp_graph_transformer import HypGraphTransformer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tier2")

METHODS = [
    ("Quadratic", {"attn_type": "quadratic"}),
    ("LorentzLinear", {"attn_type": "lorentzlinear", "use_orf": False}),
    ("LorentzLinear+ORF", {"attn_type": "lorentzlinear", "use_orf": True}),
    ("Hypformer", {"attn_type": "hypformer"}),
]

DATASET_HPARAMS = {
    "disease_nc": {"d_hyp": 32, "lr": 1e-3, "epochs": 500, "dropout": 0.1,
                   "M": 64, "n_layers": 2, "wd": 1e-4},
    "cora":       {"d_hyp": 32, "lr": 1e-3, "epochs": 500, "dropout": 0.3,
                   "M": 128, "n_layers": 2, "wd": 1e-4},
    "citeseer":   {"d_hyp": 32, "lr": 1e-3, "epochs": 500, "dropout": 0.3,
                   "M": 128, "n_layers": 2, "wd": 1e-4},
    "airport":    {"d_hyp": 32, "lr": 1e-3, "epochs": 500, "dropout": 0.1,
                   "M": 64, "n_layers": 2, "wd": 1e-4},
}


def train_eval(data, method_kwargs, hparams, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HypGraphTransformer(
        d_in=data["features"].shape[1],
        d_hyp=hparams["d_hyp"],
        n_classes=data["n_classes"],
        n_layers=hparams["n_layers"],
        R=2, M=hparams["M"], beta=1.0, K=-1.0,
        dropout=hparams["dropout"],
        **method_kwargs,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams["lr"], weight_decay=hparams["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hparams["epochs"], eta_min=1e-5)

    features = data["features"]
    labels = data["labels"]
    train_mask, val_mask, test_mask = data["train_mask"], data["val_mask"], data["test_mask"]

    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_counter = 0

    for epoch in range(hparams["epochs"]):
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(features)
                pred = logits.argmax(dim=-1)
                val_acc = (pred[val_mask] == labels[val_mask]).float().mean().item()
                test_acc = (pred[test_mask] == labels[test_mask]).float().mean().item()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= 20:
                    break

    return best_test_acc, best_val_acc


def run(datasets=None, n_seeds=5):
    if datasets is None:
        datasets = ["disease_nc", "cora", "citeseer", "airport"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*70}")

        data = load_dataset(ds_name)
        hparams = DATASET_HPARAMS[ds_name]
        print(f"  N={data['n_nodes']}, d={data['features'].shape[1]}, "
              f"classes={data['n_classes']}")
        print(f"  hparams: d_hyp={hparams['d_hyp']}, lr={hparams['lr']}, "
              f"M={hparams['M']}, layers={hparams['n_layers']}, "
              f"dropout={hparams['dropout']}")

        ds_results = {}
        print(f"\n  {'Method':<22} {'Test (mean±std)':>18} {'Val (mean)':>12} {'Time':>8}")
        print(f"  {'-'*62}")

        for method_name, method_kwargs in METHODS:
            accs, val_accs = [], []
            t0 = time.time()
            for seed in range(n_seeds):
                test_acc, val_acc = train_eval(data, method_kwargs, hparams, seed=seed)
                accs.append(test_acc)
                val_accs.append(val_acc)
            elapsed = time.time() - t0

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_val = np.mean(val_accs)
            ds_results[method_name] = {
                "test_mean": float(mean_acc),
                "test_std": float(std_acc),
                "val_mean": float(mean_val),
                "runs": [float(a) for a in accs],
            }
            print(f"  {method_name:<22} {mean_acc:>7.1%} ± {std_acc:>5.1%}  "
                  f"{mean_val:>10.1%}  {elapsed:>6.1f}s")

        all_results[ds_name] = ds_results

    with open(os.path.join(RESULTS_DIR, "benchmark_results_tuned.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    _plot_results(all_results)
    return all_results


def _plot_results(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(all_results.keys())
    methods = [m[0] for m in METHODS]
    colors = {"Quadratic": "#1b9e77", "LorentzLinear": "#d95f02",
              "LorentzLinear+ORF": "#7570b3", "Hypformer": "#e7298a"}

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        ds = all_results[ds_name]
        x = np.arange(len(methods))
        means = [ds[m]["test_mean"] for m in methods]
        stds = [ds[m]["test_std"] for m in methods]
        bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.7,
                      color=[colors[m] for m in methods], alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.01, f"{m:.1%}", ha="center", fontsize=8,
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Test Accuracy")
        ax.set_title(ds_name, fontsize=11, fontweight="bold")
        ymax = max(means) + max(stds)
        ax.set_ylim(0, min(1.0, ymax * 1.3))
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Tier 2 Node Classification (tuned, 5 seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "tier2_benchmarks_tuned.png"), dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved to {RESULTS_DIR}/tier2_benchmarks_tuned.png")


if __name__ == "__main__":
    run()

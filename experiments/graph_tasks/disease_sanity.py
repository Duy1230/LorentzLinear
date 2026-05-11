"""Disease-like sanity check: node classification on a tree-structured graph.

Uses a synthetic disease-like tree (1,044 nodes, tree-structured, hierarchical)
to test whether LorentzLinear's ~18% kernel error destroys downstream accuracy
relative to quadratic attention.

This is a de-risking experiment for Tier 2: if LorentzLinear matches or beats
quadratic on this hierarchical task, the full Tier 2 benchmarks are worth running.
"""

from __future__ import annotations

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.hyp_graph_transformer import HypGraphTransformer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tier2")


def _generate_synthetic_tree(n_nodes: int = 1044, n_classes: int = 3,
                             d_feat: int = 8, max_children: int = 4,
                             seed: int = 42, hard: bool = False):
    """Generate a synthetic tree dataset.

    When hard=False: original shallow tree with depth-banded labels.
    When hard=True: deep imbalanced tree (depth 12-15) with path-dependent
    labels that require full hierarchical understanding.
    """
    rng = np.random.RandomState(seed)

    if hard:
        return _generate_hard_tree(n_nodes, n_classes, d_feat, rng)

    adj = []
    depths = [0]
    queue = [0]
    node_count = 1

    while queue and node_count < n_nodes:
        parent = queue.pop(0)
        n_children = rng.randint(1, max_children + 1)
        for _ in range(n_children):
            if node_count >= n_nodes:
                break
            child = node_count
            adj.append((parent, child))
            adj.append((child, parent))
            depths.append(depths[parent] + 1)
            queue.append(child)
            node_count += 1

    depths = np.array(depths[:n_nodes])
    max_depth = depths.max()

    features = np.zeros((n_nodes, d_feat), dtype=np.float32)
    for i in range(n_nodes):
        depth_signal = depths[i] / (max_depth + 1)
        features[i] = rng.randn(d_feat) * 0.3 + depth_signal

    boundaries = np.linspace(0, max_depth + 1, n_classes + 1)
    labels = np.digitize(depths, boundaries[1:])
    labels = np.clip(labels, 0, n_classes - 1)

    n_train = int(0.6 * n_nodes)
    n_val = int(0.2 * n_nodes)
    perm = rng.permutation(n_nodes)
    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    return {
        "features": torch.tensor(features),
        "labels": torch.tensor(labels, dtype=torch.long),
        "train_mask": torch.tensor(train_mask),
        "val_mask": torch.tensor(val_mask),
        "test_mask": torch.tensor(test_mask),
        "n_classes": n_classes,
        "n_nodes": n_nodes,
        "max_depth": int(max_depth),
    }


def _generate_hard_tree(n_nodes: int, n_classes: int, d_feat: int, rng):
    """Deep imbalanced tree (depth 12-15) with path-dependent labels.

    Phase 1: grow n_classes spines from root to depth=target_depth using
    chain+rare-branch growth (branching probability 20%).
    Phase 2: fill remaining nodes as leaves of existing deep nodes using
    moderate branching (capped at max_depth).

    Labels = major subtree index (which depth-1 branch the node belongs to).
    Features: noisy path encoding at each branching level, plus local
    structure signals.  Single-node features are weakly discriminative;
    the task rewards attention-based aggregation across the subtree.
    """
    target_depth = 14
    max_depth_cap = target_depth + 2
    parents = [-1]
    depths = [0]
    branch_ids = [0]
    node_count = 1

    spine_heads = []
    for c in range(min(n_classes, n_nodes - 1)):
        parents.append(0)
        depths.append(1)
        branch_ids.append(c)
        spine_heads.append(node_count)
        node_count += 1

    # Phase 1: grow spines to target_depth
    for head in spine_heads:
        cur = head
        for level in range(2, target_depth + 1):
            if node_count >= n_nodes:
                break
            if rng.rand() < 0.20 and node_count + 1 < n_nodes:
                for _ in range(2):
                    if node_count >= n_nodes:
                        break
                    parents.append(cur)
                    depths.append(level)
                    branch_ids.append(branch_ids[cur])
                    node_count += 1
                cur = node_count - 1
            else:
                parents.append(cur)
                depths.append(level)
                branch_ids.append(branch_ids[cur])
                cur = node_count
                node_count += 1

    # Phase 2: fill remaining nodes as children of existing deep nodes
    deep_nodes = [i for i in range(node_count) if depths[i] >= target_depth - 3]
    while node_count < n_nodes:
        if not deep_nodes:
            deep_nodes = list(range(1, node_count))
        parent = deep_nodes[rng.randint(0, len(deep_nodes))]
        d = depths[parent] + 1
        if d > max_depth_cap:
            deep_nodes.remove(parent)
            continue
        n_ch = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        for _ in range(n_ch):
            if node_count >= n_nodes:
                break
            parents.append(parent)
            depths.append(d)
            branch_ids.append(branch_ids[parent])
            node_count += 1

    depths = np.array(depths[:n_nodes])
    branch_ids = np.array(branch_ids[:n_nodes])
    max_depth = depths.max()

    children_count = np.zeros(n_nodes)
    for i, p in enumerate(parents):
        if p >= 0:
            children_count[p] += 1
    subtree_sizes = np.ones(n_nodes)
    for i in range(n_nodes - 1, 0, -1):
        subtree_sizes[parents[i]] += subtree_sizes[i]

    # Features: noisy path encoding + local structure
    # Assign each depth-1 branch a random direction vector; children inherit
    # a noisy version, creating a hierarchical embedding in feature space.
    branch_dirs = rng.randn(n_classes, d_feat).astype(np.float32) * 0.5

    features = np.zeros((n_nodes, d_feat), dtype=np.float32)
    for i in range(n_nodes):
        bid = branch_ids[i] % n_classes
        d_norm = depths[i] / (max_depth + 1)
        c_norm = children_count[i] / max(children_count.max(), 1)
        s_norm = np.log1p(subtree_sizes[i]) / np.log1p(subtree_sizes[0])

        path_signal = branch_dirs[bid] * (1.0 - 0.3 * d_norm)
        features[i] = path_signal + rng.randn(d_feat) * 0.6
        features[i, 0] += d_norm * 0.3
        features[i, 1] += c_norm * 0.2
        features[i, 2] += s_norm * 0.2

    labels = branch_ids % n_classes

    n_train = int(0.6 * n_nodes)
    n_val = int(0.2 * n_nodes)
    perm = rng.permutation(n_nodes)
    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    return {
        "features": torch.tensor(features),
        "labels": torch.tensor(labels.astype(np.int64)),
        "train_mask": torch.tensor(train_mask),
        "val_mask": torch.tensor(val_mask),
        "test_mask": torch.tensor(test_mask),
        "n_classes": n_classes,
        "n_nodes": n_nodes,
        "max_depth": int(max_depth),
    }


def _train_eval(attn_type, data, d_hyp=16, lr=1e-3, epochs=200,
                R=2, M=64, use_orf=False, seed=0):
    """Train a HypGraphTransformer and return test accuracy."""
    torch.manual_seed(seed)
    model = HypGraphTransformer(
        d_in=data["features"].shape[1],
        d_hyp=d_hyp,
        n_classes=data["n_classes"],
        attn_type=attn_type,
        R=R, M=M, beta=1.0, K=-1.0,
        use_orf=use_orf,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    features = data["features"]
    labels = data["labels"]
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(features)
                pred = logits.argmax(dim=-1)
                val_acc = (pred[val_mask] == labels[val_mask]).float().mean().item()
                test_acc = (pred[test_mask] == labels[test_mask]).float().mean().item()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc

    return best_test_acc, best_val_acc


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    n_nodes = cfg.get("n_nodes", 1044)
    n_classes = cfg.get("n_classes", 3)
    d_feat = cfg.get("d_feat", 8)
    d_hyp = cfg.get("d_hyp", 16)
    epochs = cfg.get("epochs", 200)
    n_seeds = cfg.get("n_seeds", 5)
    lr = cfg.get("lr", 1e-3)
    hard = cfg.get("hard", False)

    data = _generate_synthetic_tree(
        n_nodes=n_nodes, n_classes=n_classes, d_feat=d_feat, hard=hard)
    mode = "HARD (path-dependent)" if hard else "EASY (depth-banded)"
    print(f"  Generated {mode} tree: {data['n_nodes']} nodes, "
          f"max_depth={data['max_depth']}, {data['n_classes']} classes")

    configs = [
        ("quadratic", {"attn_type": "quadratic"}),
        ("lorentzlinear_iid", {"attn_type": "lorentzlinear", "use_orf": False}),
        ("lorentzlinear_orf", {"attn_type": "lorentzlinear", "use_orf": True}),
        ("hypformer", {"attn_type": "hypformer"}),
    ]

    results = {}
    print(f"\n  {'Method':<22}  {'Test Acc (mean)':>14}  {'Std':>8}  {'Seeds':>6}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*8}  {'-'*6}")

    for name, kwargs in configs:
        accs = []
        for seed in range(n_seeds):
            t0 = time.time()
            test_acc, val_acc = _train_eval(
                data=data, d_hyp=d_hyp, lr=lr, epochs=epochs,
                seed=seed, **kwargs,
            )
            accs.append(test_acc)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results[name] = {"mean": mean_acc, "std": std_acc, "runs": accs}
        print(f"  {name:<22}  {mean_acc:>13.1%}  {std_acc:>7.1%}  {n_seeds:>6}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
                  color=colors[:len(names)], alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Test Accuracy")
    tag = "hard" if hard else "easy"
    ax.set_title(f"Synthetic Tree ({tag})  ({n_nodes} nodes, depth≤{data['max_depth']}, {n_seeds} seeds)")
    ax.set_ylim(0, 1.05)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"disease_sanity_{tag}.png"), dpi=150)
    plt.close(fig)

    print(f"\n  Plot saved to {os.path.join(RESULTS_DIR, 'disease_sanity.png')}")
    return results


if __name__ == "__main__":
    run()

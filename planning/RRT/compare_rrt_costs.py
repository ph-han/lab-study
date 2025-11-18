# Created with help from Codex (GPT-5 assistant)

import sys
from pathlib import Path
import importlib.util
from statistics import mean, median, pstdev

import matplotlib.pyplot as plt

# Ensure local modules can be imported
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import rrt


def load_rrt_star_module():
    """Import rrt-star.py despite the hyphen in its file name."""
    spec = importlib.util.spec_from_file_location("rrt_star_module", CURRENT_DIR / "rrt-star.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rrt_star = load_rrt_star_module()


def collect_costs(num_runs=100, iter_num=1000, base_seed=0, seed_gap=10):
    seeds = [base_seed + i * seed_gap for i in range(num_runs)]
    rrt_costs = []
    rrt_star_costs = []

    total = len(seeds)

    for i, seed in enumerate(seeds, start=1):
        rrt_final, _ = rrt.run_rrt(iter_num=iter_num, seed=seed, do_plot=False)
        rrt_costs.append(rrt_final.cost)

        rrt_star_final, _ = rrt_star.run_rrt_star(seed=seed, do_plot=False)
        rrt_star_costs.append(rrt_star_final.cost)

        # --- print progress bar ---
        progress = i / total
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {i}/{total}")
        sys.stdout.flush()

    print() 

    return seeds, rrt_costs, rrt_star_costs


def compute_stats(costs):
    return {
        "mean": mean(costs),
        "median": median(costs),
        "min": min(costs),
        "max": max(costs),
        "std": pstdev(costs) if len(costs) > 1 else 0.0,
    }


def format_stats(label, stats):
    return (
        f"{label}: mean={stats['mean']:.2f}, "
        f"median={stats['median']:.2f}, "
        f"min={stats['min']:.2f}, max={stats['max']:.2f}, "
        f"std={stats['std']:.2f}"
    )


def plot_costs(seeds, rrt_costs, rrt_star_costs, rrt_stats, rrt_star_stats):
    plt.figure()
    plt.plot(seeds, rrt_costs, "o-", label="RRT")
    plt.plot(seeds, rrt_star_costs, "s-", label="RRT*")
    plt.xlabel("Random seed")
    plt.ylabel("Final path cost")
    plt.title("RRT vs RRT* final costs per seed")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    stats_text = "\n".join(
        [format_stats("RRT ", rrt_stats), format_stats("RRT*", rrt_star_stats)]
    )
    plt.gcf().text(
        0.02,
        0.95,
        stats_text,
        va="center_baseline",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()


def plot_histograms(rrt_costs, rrt_star_costs, rrt_stats, rrt_star_stats, bins=15):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    fig.suptitle("Distribution of final path lengths", fontsize=12)
    data = [
        ("RRT*", rrt_star_costs, rrt_star_stats, "tab:green"),
        ("RRT", rrt_costs, rrt_stats, "tab:red"),
    ]

    for ax, (label, costs, stats, color) in zip(axes, data):
        ax.hist(costs, bins=bins, color=color, alpha=0.85, edgecolor="black")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.text(
            0.65,
            0.8,
            f"Mean = {stats['mean']:.2f}\nStd dev = {stats['std']:.2f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # show RRT* min/max range on RRT axis to mimic reference dashed lines
    axes[-1].axvline(rrt_star_stats["min"], linestyle="--", color="k", linewidth=1)
    axes[-1].axvline(rrt_star_stats["max"], linestyle="--", color="k", linewidth=1)
    axes[-1].set_xlabel("Path length (cost)")
    axes[-1].set_xlim(
        min(min(rrt_costs), min(rrt_star_costs)) - 1,
        max(max(rrt_costs), max(rrt_star_costs)) + 1,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    seeds, rrt_costs, rrt_star_costs = collect_costs(num_runs=1000)
    print("Seeds      :", seeds)
    print("RRT costs  :", rrt_costs)
    print("RRT* costs :", rrt_star_costs)

    rrt_stats = compute_stats(rrt_costs)
    rrt_star_stats = compute_stats(rrt_star_costs)

    print(format_stats("RRT ", rrt_stats))
    print(format_stats("RRT*", rrt_star_stats))

    plot_costs(seeds, rrt_costs, rrt_star_costs, rrt_stats, rrt_star_stats)
    plot_histograms(rrt_costs, rrt_star_costs, rrt_stats, rrt_star_stats)

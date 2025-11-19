"""
Compare RRT and RRT* on a single seed and plot their paths on one figure.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Ensure the local planning/RRT package is discoverable
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import rrt  # pylint: disable=wrong-import-position


def load_rrt_star_module():
    """Import rrt-star.py despite the hyphen in its file name."""
    spec = importlib.util.spec_from_file_location("rrt_star_module", CURRENT_DIR / "rrt-star.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rrt_star = load_rrt_star_module()


def extract_path_coords(final_node):
    """Return x/y lists for plotting from the linked parent chain."""
    if final_node is None:
        return [], []

    xs, ys = [], []
    curr = final_node
    while curr is not None:
        xs.append(curr.x)
        ys.append(curr.y)
        curr = curr.parent

    xs.reverse()
    ys.reverse()
    return xs, ys


def compare_single_seed(seed=0, iter_num=1000, show_plot=True):
    """
    Run RRT and RRT* with the same seed once and plot both paths.

    Parameters
    ----------
    seed : int
        Random seed shared by both planners.
    iter_num : int
        Iteration count passed to the classic RRT implementation.
    show_plot : bool
        When True, display the matplotlib figure; useful to disable for tests.
    """
    final_rrt, rrt_solver = rrt.run_rrt(iter_num=iter_num, seed=seed, do_plot=False)
    final_rrt_star, rrt_star_solver = rrt_star.run_rrt_star(seed=seed, do_plot=False, is_rewiring=False)

    if final_rrt is None or final_rrt_star is None:
        raise RuntimeError("One of the planners failed to return a final node.")

    rrt_path = extract_path_coords(final_rrt)
    rrt_star_path = extract_path_coords(final_rrt_star)

    plt.figure()
    rrt_solver.cspace.plot()
    plt.plot(rrt_solver.init_x, rrt_solver.init_y, "ob", label="Start")
    plt.plot(rrt_solver.goal_x, rrt_solver.goal_y, "xr", label="Goal")
    plt.plot(rrt_path[0], rrt_path[1], "-r", linewidth=1.5, alpha=0.5, label=f"RRT (cost: {final_rrt.cost:.2f})")
    plt.plot(
        rrt_star_path[0],
        rrt_star_path[1],
        "-b",
        linewidth=1.5,
        alpha=0.5,
        label=f"RRT* (cost: {final_rrt_star.cost:.2f})",
    )

    plt.title(f"RRT vs RRT* (seed={seed})")
    plt.axis("equal")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    print(f"Seed: {seed}")
    print(f"RRT  cost: {final_rrt.cost:.3f} (goal reached: {rrt_solver.is_goal})")
    print(f"RRT* cost: {final_rrt_star.cost:.3f} (goal reached: {rrt_star_solver.is_goal})")

    if show_plot:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Run RRT and RRT* once with the same seed and plot both results.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed shared by both planners.")
    parser.add_argument(
        "--iter-num",
        type=int,
        default=1000,
        help="Iteration count passed to the classic RRT run_rrt helper.",
    )
    parser.add_argument("--no-show", action="store_true", help="Skip displaying the matplotlib window.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_single_seed(seed=args.seed, iter_num=args.iter_num, show_plot=not args.no_show)

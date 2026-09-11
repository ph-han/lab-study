import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from neural_rrt_star2 import NeuralRRTStar, set_seed
from train_recon2 import NeuralRRTStarNet


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralRRTStarNet().to(device)
    state = torch.load("best_neural_rrt_star_net_iou_2.pth", map_location="cpu")
    if next(iter(state)).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    seeds = list(range(0, 1000))
    map_path = "./dataset/test/maps/000726.png"

    for ss in tqdm([6, 4, 2, 1], desc="Testing step size..."):
        neural_stats = {"nodes": [], "cost": [], "time": [], "success": 0}
        rrt_stats = {"nodes": [], "cost": [], "time": [], "success": 0}

        for seed in tqdm(seeds, desc="Testing..."):
            set_seed(42 + seed)

            # --- Neural RRT* ---
            neural_planner = NeuralRRTStar(42, 2, ss, 7000, map_path,
                                           is_neural_mode=True, expand_size=ss)
            neural_node, tot_time_neural = neural_planner.planning(model=model, device=device, is_rewiring=True, is_break=True)

            if neural_node:
                neural_stats["success"] += 1
                neural_stats["nodes"].append(len(neural_planner.paths))
                neural_stats["cost"].append(neural_node.cost)
                neural_stats["time"].append(tot_time_neural)

            # --- RRT* ---
            set_seed(1000 + seed)
            rrt_planner = NeuralRRTStar(42, 2, ss, 7000, map_path,
                                        is_neural_mode=False, expand_size=ss)
            rrt_node, tot_time_rrt = rrt_planner.planning(is_rewiring=True, is_break=True)

            if rrt_node:
                rrt_stats["success"] += 1
                rrt_stats["nodes"].append(len(rrt_planner.paths))
                rrt_stats["cost"].append(rrt_node.cost)
                rrt_stats["time"].append(tot_time_rrt)

        # --- 결과 요약 ---
        def mean_safe(lst):
            return np.mean(lst) if len(lst) else np.nan

        metrics = ["#Nodes", "Cost", "Time(s)", "Success(%)"]
        neural_means = [
            mean_safe(neural_stats["nodes"]),
            mean_safe(neural_stats["cost"]),
            mean_safe(neural_stats["time"]),
            100 * neural_stats["success"] / len(seeds),
        ]
        rrt_means = [
            mean_safe(rrt_stats["nodes"]),
            mean_safe(rrt_stats["cost"]),
            mean_safe(rrt_stats["time"]),
            100 * rrt_stats["success"] / len(seeds),
        ]

        # --- 시각화 ---
        labels = ["Neural RRT*", "RRT*"]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        fig.suptitle(f"Comparison at Step Size={ss}", fontsize=14)

        colors = ["tab:blue", "tab:orange"]
        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.bar(x[0], neural_means[i], color=colors[0])
            ax.bar(x[1], rrt_means[i], color=colors[1])
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(axis='y')
            if i == 0:
                ax.legend(labels)
        plt.tight_layout()
        plt.savefig(f"result-step-size-{ss}.png")
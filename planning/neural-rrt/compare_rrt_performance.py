import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

# 모델 정의 import
from train_recon2 import NeuralRRTStarNet

# 두 가지 버전의 Planner를 별칭(Alias)을 사용하여 import
from neural_rrt_star import NeuralRRTStar as OldRRTStar
from neural_rrt_star2 import NeuralRRTStar as NewRRTStar
from neural_rrt_star2 import set_seed

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def main():
    # --- 설정 (Configuration) ---
    # 테스트에 사용할 맵 경로 (Context에 있는 경로 중 하나 선택)
    MAP_PATH = "./dataset/test/maps/000455.png" 
    # 학습된 가중치 파일 경로
    WEIGHTS_PATH = "best_neural_rrt_star_net_iou2.pth"
    
    CLEARANCES = [1, 2, 4, 6]
    NUM_SEEDS = 1
    ITER_MAX = 7000
    STEP_SIZE = 2
    EXPAND_SIZE = 2
    
    # --- 초기화 (Setup) ---
    device = get_device()
    print(f"Using device: {device}")
    
    # 모델 로드
    model = NeuralRRTStarNet().to(device)
    if os.path.exists(WEIGHTS_PATH):
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
        if next(iter(state)).startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Loaded weights from {WEIGHTS_PATH}")
    else:
        print(f"Warning: Weights file {WEIGHTS_PATH} not found. Using random weights.")
    
    model.eval()

    # --- 데이터 수집 (Data Collection) ---
    avg_times_old = []
    avg_times_new = []

    print(f"Starting comparison: {NUM_SEEDS} seeds per clearance...")

    for clr in tqdm(CLEARANCES, desc="Clearance Loop"):
        times_old = []
        times_new = []
        
        for i in tqdm(range(NUM_SEEDS), desc=f"Clr {clr} Seeds", leave=False):
            seed = 15  # 시드값 변경
            
            # 1. Old RRT* (neural_rrt_star.py)
            set_seed(seed)
            old_planner = OldRRTStar(
                seed=seed, 
                clearance=clr, 
                step_size=STEP_SIZE, 
                iter_num=ITER_MAX, 
                cspace_img_path=MAP_PATH, 
                is_neural_mode=True, 
                expand_size=EXPAND_SIZE
            )
            # 속도 측정을 위해 is_draw=False 설정
            _, time_old = old_planner.planning(model=model, device=device, is_rewiring=True, is_break=True, is_draw=False)
            times_old.append(time_old)

            # 2. New RRT* (neural_rrt_star2.py)
            set_seed(seed)
            new_planner = NewRRTStar(
                seed=seed, 
                clearance=clr, 
                step_size=STEP_SIZE, 
                iter_num=ITER_MAX, 
                cspace_img_path=MAP_PATH, 
                is_neural_mode=True, 
                expand_size=EXPAND_SIZE
            )
            _, time_new = new_planner.planning(model=model, device=device, is_rewiring=True, is_break=True, is_draw=False)
            times_new.append(time_new)

        # 평균 시간 계산
        avg_times_old.append(np.mean(times_old))
        avg_times_new.append(np.mean(times_new))

    # --- 시각화 (Visualization) ---
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(CLEARANCES))
    width = 0.35

    plt.bar(x - width/2, avg_times_old, width, label='Old Neural RRT* (neural_rrt_star.py)', color='tab:orange')
    plt.bar(x + width/2, avg_times_new, width, label='New Neural RRT* (neural_rrt_star2.py)', color='tab:blue')
    
    plt.title(f"Performance Comparison: Old vs New Neural RRT*", fontsize=14)
    plt.xlabel("Clearance", fontsize=12)
    plt.ylabel("Average Execution Time (sec)", fontsize=12)
    plt.xticks(x, CLEARANCES)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    
    output_filename = "comparison_performance_graph.png"
    plt.savefig(output_filename)
    print(f"Graph saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()
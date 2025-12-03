from train import NeuralRRTStarNet, NeuralRRTStarDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = NeuralRRTStarNet().to(device)
model.load_state_dict(torch.load("best_neural_rrt_star_net.pth"))
model.eval()

#transform
resize = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
# map_path = "./dataset/test/maps/custom_map.png"
# map_image = Image.open(map_path).convert('L')
# map_image = resize(map_image)
# map_numpy = np.array(map_image)
# h, w = map_numpy.shape
# rgb = np.ones((h, w, 3), dtype=np.float32)

# rgb[map_numpy == 1] = [0.0, 0.0, 0.0]
# rgb[map_numpy == 2] = [1.0, 0.0, 0.0]
# rgb[map_numpy == 3] = [0.0, 1.0, 0.0]

# rgb = rgb.transpose(2, 0, 1)
# map_tensor = torch.from_numpy(rgb)
# sc_tensor = torch.tensor([1, 2], dtype=torch.float32)
# input_map = map_tensor
# input_sc = sc_tensor


test_dataset = NeuralRRTStarDataset(split="test", transform=resize)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

for batch_idx, data in enumerate(tqdm(test_dataloader, desc="Testing..")):
    input_map = data['input_map'].to(device)   # [B,3,H,W]
    input_sc  = data['input_sc'].to(device)

    with torch.no_grad():
        output = model(input_map, input_sc)    # [B,1,H,W]

    B = input_map.size(0)
    for b in range(B):
        global_idx = batch_idx * B + b   # ← 전체 dataset 기준 인덱스

        im  = input_map[b].cpu()         # [3,H,W]
        out = output[b].cpu()            # [1,H,W]

        im_np  = im.permute(1, 2, 0).numpy()
        out_np = out.squeeze().numpy()

        alpha = out_np.copy()
        alpha[out_np < 1e-3] = 0.0
        alpha = np.clip(alpha * 2.0, 0.0, 0.8)

        plt.clf()
        plt.imshow(im_np, interpolation='nearest')
        plt.imshow(out_np, cmap='plasma', alpha=alpha, interpolation='bilinear')

        # start/goal 찍기
        start_mask = (im_np[...,0] == 1.0) & (im_np[...,1] == 0.0) & (im_np[...,2] == 0.0)
        goal_mask  = (im_np[...,0] == 0.0) & (im_np[...,1] == 1.0) & (im_np[...,2] == 0.0)

        start_coords = np.argwhere(start_mask)
        goal_coords  = np.argwhere(goal_mask)

        if len(start_coords) > 0:
            y, x = start_coords[0]
            plt.plot(x, y, 'o', markersize=10, color='red')

        if len(goal_coords) > 0:
            y, x = goal_coords[0]
            plt.plot(x, y, 'o', markersize=10, color='lime')

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"./result/{global_idx:04d}.png")


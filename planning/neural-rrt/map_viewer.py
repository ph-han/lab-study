import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from train import NeuralRRTStarDataset  # 파일 이름에 맞게 수정


def tensor_to_rgb_img(t):
    """[3,H,W] 텐서를 [H,W,3] numpy(0~1)로 변환"""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    if t.dim() == 3:
        t = t.permute(1, 2, 0)  # C,H,W -> H,W,C
    return t.numpy()


def tensor_to_gray_img(t):
    """[1,H,W] 텐서를 [H,W] numpy(0~1)로 변환"""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    if t.dim() == 3:
        t = t.squeeze(0)
    return t.numpy()


def make_viewer(dataset):
    idx = 0
    num_samples = len(dataset)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.tight_layout()

    def draw_sample(i):
        axes[0].cla()
        axes[1].cla()
        axes[2].cla()

        sample = dataset[i]
        input_map = sample["input_map"]      # [3,H,W]
        input_sc  = sample["input_sc"]       # [2]
        gt        = sample["gt"]             # [1,H,W]

        rgb = tensor_to_rgb_img(input_map)   # [H,W,3]
        gt_img = tensor_to_gray_img(gt)      # [H,W]

        # (0) 메타 정보 콘솔에 출력
        print(f"Index: {i} | clearance={input_sc[0].item():.3f}, step_size={input_sc[1].item():.3f}")

        # (1) RGB 맵
        axes[0].imshow(rgb, interpolation="nearest")
        axes[0].set_title(f"Input map (idx={i})")
        axes[0].axis("off")

        # (2) GT 경로만
        axes[1].imshow(gt_img, cmap="gray", interpolation="nearest")
        axes[1].set_title("GT path")
        axes[1].axis("off")

        # (3) 맵 + GT overlay
        axes[2].imshow(rgb, interpolation="nearest")
        axes[2].imshow(gt_img, cmap="plasma", alpha=0.5, interpolation="bilinear")
        axes[2].set_title("Map + GT overlay")
        axes[2].axis("off")

        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx
        if event.key == "right":
            idx = (idx + 1) % num_samples
            draw_sample(idx)
        elif event.key == "left":
            idx = (idx - 1) % num_samples
            draw_sample(idx)
        elif event.key == "g":
            print(f"[SELECT] current index = {idx}")
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_sample(idx)
    plt.show()


if __name__ == "__main__":
    # transform은 네 train 코드랑 똑같이
    resize = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)

    dataset_root_path = "./dataset"
    split = "train"   # or "valid", "test"

    ds = NeuralRRTStarDataset(dataset_root_path=dataset_root_path,
                              split=split,
                              transform=resize)

    make_viewer(ds)

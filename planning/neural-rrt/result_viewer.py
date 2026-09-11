import argparse
import csv
import glob
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt


def natural_key(path: str):
    """Sort helper to keep numbered files in numeric order."""
    base = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", base)]


def load_image_paths(folder: str):
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(folder, pattern)))
    paths.sort(key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No images found under: {folder}")
    return paths


def load_meta(meta_path: str) -> List[Tuple[float, float]]:
    """Return list indexed by sample idx => (clearance, step_size)."""
    data: List[Tuple[float, float]] = []
    if not os.path.exists(meta_path):
        print(f"[WARN] meta not found: {meta_path}")
        return data
    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                clearance = float(row.get("clearance", "nan"))
                step = float(row.get("step_size", "nan"))
            except ValueError:
                clearance, step = float("nan"), float("nan")
            data.append((clearance, step))
    print(f"[INFO] loaded meta rows: {len(data)} from {meta_path}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Keyboard viewer for images in the result folder.")
    parser.add_argument(
        "--dir",
        default=os.path.join(os.path.dirname(__file__), "result"),
        help="Directory containing result images (default: ./result next to this script)",
    )
    parser.add_argument(
        "--meta",
        default=os.path.join(os.path.dirname(__file__), "dataset", "test", "meta.csv"),
        help="Path to meta.csv containing clearance/step_size info.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based).")
    args = parser.parse_args()

    image_paths = load_image_paths(args.dir)
    meta = load_meta(args.meta)
    num_images = len(image_paths)
    idx = max(0, min(args.start, num_images - 1))

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.tight_layout()

    def draw(i: int):
        ax.cla()
        img = plt.imread(image_paths[i])
        ax.imshow(img)

        meta_text = ""
        if i < len(meta):
            clearance, step = meta[i]
            meta_text = f" | clearance={clearance:.3f}, step_size={step:.3f}"
        else:
            meta_text = " | meta: N/A"

        ax.set_title(f"{os.path.basename(image_paths[i])} ({i + 1}/{num_images}){meta_text}", fontsize=10)
        ax.axis("off")
        print(f"[VIEW] index={i} file={image_paths[i]}{meta_text}")
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx
        if event.key in ("right", " "):
            idx = (idx + 1) % num_images
            draw(idx)
        elif event.key == "left":
            idx = (idx - 1) % num_images
            draw(idx)
        elif event.key == "home":
            idx = 0
            draw(idx)
        elif event.key == "end":
            idx = num_images - 1
            draw(idx)
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw(idx)
    plt.show()


if __name__ == "__main__":
    main()

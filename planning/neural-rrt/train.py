import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MapDataset(Dataset):
    def __init__(self, annotations_file, img_dir, tarnsform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        pass
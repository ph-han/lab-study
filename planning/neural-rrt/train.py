
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from resnet import ResNet50

import matplotlib.pyplot as plt


class NeuralRRTStarDataset(Dataset):
    def __init__(self, dataset_root_path="./dataset", split="train", transform=None):
        """
        Args:
            dataset_root_path (string): dataset root path
            split (string): "train", "valid", or "test"
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert split in ["train", "valid", "test"], "split must be one of 'train', 'valid', or 'test'"
        dataset_path = f"{dataset_root_path}/{split}"

        self.meta_data = pd.read_csv(dataset_path + "/meta.csv")
        self.transform = transform

        self.clearance_list = []
        self.step_size_list = []
        self.map_path_list = []
        self.gt_path_list = []

        for row in tqdm(self.meta_data.iterrows(), total=len(self.meta_data), desc=f"Loading {dataset_path} meta data..."):
            self.clearance_list.append(row[1]["clearance"])
            self.step_size_list.append(row[1]["step_size"])
            self.map_path_list.append(row[1]["map_path"])
            self.gt_path_list.append(row[1]["gt_path"])


    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        map_path = self.map_path_list[idx]
        gt_path = self.gt_path_list[idx]

        map_image = Image.open(map_path).convert('L')
        gt_image = Image.open(gt_path).convert('L')

        if self.transform:
            map_image = self.transform(map_image)
            gt_image = self.transform(gt_image)

        map_numpy = np.array(map_image)
        gt_numpy = np.array(gt_image)

        h, w = map_numpy.shape
        rgb = np.ones((h, w, 3), dtype=np.float32)


        rgb[map_numpy == 1] = [0.0, 0.0, 0.0]
        rgb[map_numpy == 2] = [1.0, 0.0, 0.0]
        rgb[map_numpy == 3] = [0.0, 1.0, 0.0]

        rgb = rgb.transpose(2, 0, 1)
        map_tensor = torch.from_numpy(rgb)

        # gt는 그대로 단일 채널
        gt_tensor = torch.from_numpy(gt_numpy).unsqueeze(0).float()
        reconst_gt = np.zeros_like(map_numpy, dtype=np.float32)
        reconst_gt[map_numpy == 1] = 1.0
        reconst_gt = torch.from_numpy(reconst_gt).unsqueeze(0).float()

        clearance = self.clearance_list[idx]
        step_size = self.step_size_list[idx]
        SC_tensor = torch.tensor([clearance, step_size], dtype=torch.float32)

        return {'input_map': map_tensor, 'input_sc': SC_tensor, 'gt': gt_tensor, 'reconst_gt': reconst_gt}

    

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)
    
        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
        
        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)


    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        return out
    
class NeuralRRTStarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = ResNet50()
        self.attribute_fc1 = nn.Linear(2, 32)
        self.attribute_fc2 = nn.Linear(32, 64)

        self.low_conv = nn.Sequential(
            nn.Conv2d(256 + 32, 256 + 32, kernel_size=3, dilation=1, stride=1, padding=1),
            nn.BatchNorm2d(256 + 32),
            nn.ReLU(inplace=True)
        )
        self.aspp = ASPP(2048, 256)

        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(608, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.reconst_conv = nn.Conv2d(64, 1, kernel_size=1)

    def attribute_encode(self, x):
        fc1 = F.relu(self.attribute_fc1(x))
        fc2 = F.relu(self.attribute_fc2(fc1))

        low = fc1.view(-1, 32, 1, 1)
        high = fc2.view(-1, 64, 1, 1)

        return low, high

    def forward(self, input_map, input_attr):
        # ----- encode -----
        c1, c4 = self.resnet50(input_map) # [16, 256, 28, 28], [16, 2048, 7, 7]
        f_al, f_ah = self.attribute_encode(input_attr) # [16, 32, 1, 1], [16, 64, 1, 1]

        f_l = c1
        f_h = self.aspp(c4)

        f_al  = f_al.expand(-1, -1, f_l.size(2), f_l.size(3)) # [16, 32, 28, 28]
        f_ah = f_ah.expand(-1, -1, f_h.size(2), f_h.size(3)) # [16, 64, 7, 7]

        fc_low = self.low_conv(torch.cat([f_l, f_al], dim=1))
        fc_high= torch.cat([f_h, f_ah], dim=1)
        fc_high_up = F.interpolate(fc_high, size=(fc_low.size(2), fc_low.size(3)), mode="bilinear", align_corners=False)

        f_e = torch.cat([fc_low, fc_high_up], dim=1)

        # ----- decode -----
        f_d = self.dec_conv1(f_e)
        f_d = F.interpolate(f_d, scale_factor=2, mode="bilinear", align_corners=False)

        f_d = self.dec_conv2(f_d)
        f_d = F.interpolate(f_d, scale_factor=2, mode="bilinear", align_corners=False)

        f_d = self.dec_conv3(f_d)
        f_d = F.interpolate(f_d, size=(input_map.size(2), input_map.size(3)),
                            mode="bilinear", align_corners=False)
        
        out = F.sigmoid(self.out_conv(f_d))
        reconst = torch.sigmoid(self.reconst_conv(f_d))
        return out, reconst
        
def eval(net, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    loss_fn = nn.BCELoss()

    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    count = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in loop:
            input_map = batch['input_map'].to(device)
            input_sc  = batch['input_sc'].to(device)
            gt        = batch['gt'].to(device)
            reconst_gt = batch['reconst_gt'].to(device)


            out, reconst = net(input_map, input_sc)
            loss = loss_fn(out, gt) + 0.5 * loss_fn(reconst, reconst_gt)
            total_loss += loss.item()

            pred = out
            binary = (pred > 0.5).float()

            inter = (binary * gt).sum(dim=(1,2,3))
            union = ((binary + gt) > 0).float().sum(dim=(1,2,3))
            iou = (inter / (union + 1e-6)).mean().item()
            total_iou += iou

            acc = (binary == gt).float().mean().item()
            total_acc += acc

            count += 1
            loop.set_postfix(loss=loss.item(), IoU=iou, Acc=acc)

    avg_loss = total_loss / count
    avg_iou = total_iou / count
    avg_acc = total_acc / count

    print(f"\n✅ Eval | Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Acc: {avg_acc:.4f}")
    return avg_loss, avg_iou, avg_acc

def train(dataloader, net, epoch, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    optimizer = AdamW(net.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    net.train()
    
    loss_list = []
    running_loss = 0.0
    loop = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=False
        )
    for step, batch in enumerate(loop):
        input_map = batch['input_map'].to(device)
        input_sc = batch['input_sc'].to(device)
        gt = batch['gt'].to(device)
        reconst_gt = batch['reconst_gt'].to(device)

        optimizer.zero_grad()

        output, reconst = net(input_map, input_sc)
        loss = loss_fn(output, gt) + 0.5 * loss_fn(reconst, reconst_gt)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}/{num_epochs} | Step [{step+1}/{len(dataloader)}]")
        loop.set_postfix(loss=loss.item())
        if (step + 1) % 10 == 0:
            avg_loss = running_loss / 10
            running_loss = 0.0
            print(f"{step + 1} Loss: {avg_loss:.4f}")
            loss_list.append(avg_loss)

if __name__ == "__main__":
    #transform
    resize = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)

    # dataset
    train_dataset = NeuralRRTStarDataset(split="train", transform=resize)
    valid_dataset = NeuralRRTStarDataset(split="valid", transform=resize)
    # test_dataset = NeuralRRTStarDataset(split="test", transform=resize)

    # print(len(train_dataset))
    # it = iter(train_dataset)
    # print(next(it)['input_map'].shape)
    # print(next(it)['input_sc'].shape)
    # print(next(it)['gt'].shape)

    # dataloader
    batch_size = 115
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # batch_iterator = iter(train_dataloader)
    # batch = next(batch_iterator)  
    # print(batch['input_map'].shape)
    # print(batch['input_sc'].shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NeuralRRTStarNet().to(device)
    # out = net(batch['input_map'], batch['input_sc'])
    # print(out.shape)
    # print("Output min/max:", out.min().item(), out.max().item())

    train_loss_list = []
    num_epochs=35
    best_loss = float('inf')
    best_iou = float('-inf')
    for epoch in range(num_epochs):
        loss_list = train(train_dataloader, net, epoch, num_epochs)
        train_loss_list.append(loss_list)
        val_loss, val_iou, val_acc = eval(net, valid_dataloader)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(net.state_dict(), "best_neural_rrt_star_net_iou.pth")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), "best_neural_rrt_star_net_loss.pth")
        print(f"[Val] Epoch {epoch+1} | Loss={val_loss:.4f}, IoU={val_iou:.4f}, Acc={val_acc:.4f}")

    # torch.save(net.state_dict(), "neural_rrt_star_net.pth")
    # print("Training finished & model saved.")

    plt.plot(train_loss_list)
    plt.show()
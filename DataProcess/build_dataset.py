import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter


class TemporalElasticTransform:
    """ 时序弹性变换（保持不变）"""

    def __init__(self, alpha=20, sigma=5, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, frames):
        if np.random.rand() > self.p:
            return frames
        displacement = np.random.uniform(-self.alpha, self.alpha, size=len(frames))
        displacement = gaussian_filter(displacement, sigma=self.sigma)
        new_frames = []
        for t in range(len(frames)):
            src_t = min(max(int(t + displacement[t]), 0), len(frames) - 1)
            new_frames.append(frames[src_t])
        return new_frames


class ATTMicroDataset(Dataset):
    def __init__(self, root_dir, phase="train", seq_length=5):
        self.root_dir = root_dir
        self.phase = phase
        self.seq_length = seq_length
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = self._load_samples()

        # 数据增强
        self.spatial_tf = self._get_spatial_transform()
        self.temporal_tf = TemporalElasticTransform() if phase == "train" else None

    def _load_samples(self):
        samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            # 获取所有.jpg文件
            all_files = [f for f in os.listdir(cls_dir) if f.endswith(".jpg")]

            # 按文件名中的时间戳排序
            try:
                # 匹配 reg_imgXX (X).jpg 或 reg_imgXX.jpg
                all_files.sort(key=lambda x: int(re.search(r"reg_img(\d+)(?:\s*\((\d+)\))?", x).group(1)))
            except Exception as e:
                print(f"警告: 无法解析文件名格式: {e}")
                continue  # 跳过无法解析的目录

            # 按seq_length分组
            num_sequences = len(all_files) // self.seq_length
            for i in range(num_sequences):
                start = i * self.seq_length
                end = start + self.seq_length
                file_group = all_files[start:end]
                samples.append((
                    (cls_dir, file_group),  # 存储文件组信息
                    self.class_to_idx[cls]
                ))
        print(f"Loaded {len(samples)} sequences from {self.root_dir}")
        return samples

    def _get_spatial_transform(self):
        return transforms.Compose([
            transforms.Resize((128, 128)),  # 修改前为 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3)
        ]) if self.phase == "train" else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_sequence(self, seq_info):
        cls_dir, file_group = seq_info
        frames = []
        for file_name in file_group:
            img_path = os.path.join(cls_dir, file_name)
            img = Image.open(img_path).convert("RGB")
            frames.append(self.spatial_tf(img))  # 形状 [C, H, W]
        # 将帧列表堆叠成一个4D张量 [C, T, H, W]
        return torch.stack(frames, dim=1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_info, label = self.samples[idx]
        seq = self._load_sequence(seq_info)

        # 确保序列形状为 [3, T, H, W]
        if seq.shape[0] != 3:
            # 如果通道数不是3，检查是否是灰度图（1通道）
            if seq.shape[0] == 1:
                # 将灰度图转换为RGB
                seq = seq.repeat(3, 1, 1, 1)
            else:
                raise ValueError(f"Input sequence must have 3 channels (RGB), but got {seq.shape[0]}")

        # 时序增强
        if self.temporal_tf:
            # 将张量转换为列表以应用时序变换
            seq_list = [seq[:, t, :, :] for t in range(seq.shape[1])]
            seq_list = self.temporal_tf(seq_list)
            # 将列表重新转换为张量
            seq = torch.stack(seq_list, dim=1)

        return seq, label

    @property
    def num_classes(self):
        return len(self.classes)

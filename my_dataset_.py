import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets.transforms import FaceKeypointBBoxSafeCrop, TemporalElasticTransform, MotionMagnification
from torchvision import transforms

class MicroExpressionDataset(Dataset):
    """微表情视频数据集
        自适应视频帧长处理
        集成时空数据增强
        支持动态人脸检测裁剪
    """
    
    def __init__(self, root_dir: str, phase: str = 'train', clip_length: int = 32, frame_size: int = 224, use_aug: bool = True):
        """
        Args:
            root_dir: 数据集根目录, root_dir/class/video_frames/
            phase: 数据[train/val/test]
            clip_length: 采样的视频片段长度
            frame_size: 输出帧的尺寸
            use_aug: 是否启用数据增强
        """
        self.root_dir = root_dir
        self.phase = phase
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.use_aug = use_aug
        
        # 初始化数据增强
        self.spatial_aug = FaceKeypointBBoxSafeCrop() if use_aug else None
        self.temporal_augs = [
            TemporalElasticTransform(p=0.5),
            MotionMagnification(factor=1.2)
        ] if use_aug else []
        
        # 加载数据集元数据
        self.classes, self.video_paths = self._scan_dataset()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 统计信息
        print(f"数据集初始化完成，共 {len(self.video_paths)} 个样本")
        print(f"类别分布: { {cls: sum(1 for p in self.video_paths if p[1]==cls) for cls in self.classes} }")


    def _scan_dataset(self):
        """
        扫描数据集目录结构
        """
        classes = sorted(os.listdir(self.root_dir))
        video_paths = []
        
        for cls in classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for video_name in os.listdir(cls_dir):
                video_dir = os.path.join(cls_dir, video_name)
                if os.path.isdir(video_dir):
                    video_paths.append((video_dir, cls))
                    
        # 打乱训练集
        if self.phase == 'train':
            random.shuffle(video_paths)
            
        return classes, video_paths


    def __len__(self):
        return len(self.video_paths)


    def __getitem__(self, idx):
        video_dir, label = self.video_paths[idx]
        frames = self._load_video_frames(video_dir)
        
        # 应用数据增强
        processed_frames = self._apply_augmentations(frames)
        
        # 转换为张量
        tensor_clip = self._frames_to_tensor(processed_frames)
        
        return tensor_clip, self.class_to_idx[label]


    def _load_video_frames(self, video_dir):
        """
        加载视频帧并预处理
        """
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith(('.jpg','.png'))])
        
        # 采样策略
        total_frames = len(frame_files)
        if total_frames >= self.clip_length:
            # 随机裁剪
            start_idx = random.randint(0, total_frames - self.clip_length)
            sampled_files = frame_files[start_idx : start_idx + self.clip_length]
        else:
            # 循环填充
            sampled_files = frame_files * (self.clip_length // total_frames + 1)
            sampled_files = sampled_files[:self.clip_length]
            
        # 加载图像
        return [Image.open(os.path.join(video_dir, f)) for f in sampled_files]


    def _apply_augmentations(self, frames):
        """应用时空数据增强"""
        # 转换为numpy数组用于处理
        np_frames = [np.array(frame) for frame in frames]
        
        # 空间增强
        if self.spatial_aug:
            np_frames = [self.spatial_aug.apply(frame) for frame in np_frames]
        
        # 时序增强
        for aug in self.temporal_augs:
            if random.random() < aug.p:
                np_frames = aug(np_frames)
                
        return [Image.fromarray(frame) for frame in np_frames]


    def _frames_to_tensor(self, frames):
        """
        将帧序列转换为视频张量[C, T, H, W]
        """
        transform = transforms.Compose([
            transforms.Resize(self.frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        tensor_frames = []
        for frame in frames:
            tensor_frames.append(transform(frame))
            
        # 调整维度为 [C, T, H, W]
        return torch.stack(tensor_frames, dim=1)

    @property
    def num_classes(self):
        return len(self.classes)


if __name__ == '__main__':
    # 测试用例
    dataset = MicroExpressionDataset(
        root_dir='path/to/dataset',
        phase='train',
        clip_length=32,
        frame_size=224
    )
    
    sample, label = dataset[0]
    print(f"样本形状: {sample.shape} | 标签: {label}")

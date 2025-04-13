import torch
from torch import nn
from torchvision.models.optical_flow import Raft_Large_Weights


class RAFTFlow(nn.Module):
    """改进的RAFT光流计算模块"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = torch.hub.load(
            "pytorch/vision",
            "raft_large",
            weights=Raft_Large_Weights.DEFAULT
        )

        # 冻结参数
        if config['FREEZE']:  # 修改为字典键访问
            for param in self.model.parameters():
                param.requires_grad_(False)

        self.flow_scale = config['FLOW_SCALE']  # 修改为字典键访问

    def forward(self, frame1, frame2):
        """
        输入：
            frame1: [B,3,H,W] 归一化到[0,1]的RGB图像
            frame2: [B,3,H,W]
        输出：
            flow: [B,2,H,W] 光流场
        """
        # 下采样到 1/2 分辨率
        # frame1 = F.interpolate(frame1, scale_factor=0.5, mode="bilinear")
        # frame2 = F.interpolate(frame2, scale_factor=0.5, mode="bilinear")

        # 确保输入是RGB格式
        if frame1.shape[1] != 3 or frame2.shape[1] != 3:
            raise ValueError("Input frames must have 3 channels (RGB)")

        # 计算多尺度光流
        flow_predictions = self.model(frame1, frame2)

        # 取最终精细光流
        flow = flow_predictions[-1] * self.flow_scale
        return flow

    @torch.no_grad()
    def precompute_flows(self, video_clip):
        """预处理整个视频的光流"""
        B, T, C, H, W = video_clip.shape
        if C != 3:
            raise ValueError("Input video must have 3 channels (RGB)")

        flows = []
        for t in range(T - 1):
            flows.append(self(video_clip[:, t], video_clip[:, t + 1]))
        return torch.stack(flows, dim=1)  # [B,T-1,2,H,W]
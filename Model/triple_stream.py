import torch
from torch import nn
import torch.nn.functional as F
from .slowfast import SlowFast
from .raft import RAFTFlow
from .deform_conv import DeformConv3d
from .fushion import DynamicFusion3D


class TripleStreamNet(nn.Module):
    def __init__(self, cfg=None, num_classes=None):
        super().__init__()
        # 参数兼容性处理
        if cfg is None:
            cfg = self._get_default_config(num_classes)
        elif num_classes is not None:
            cfg["MODEL"]["NUM_CLASSES"] = num_classes

        # 初始化主干网络
        self.slowfast = SlowFast(
            slowfast_config=cfg["MODEL"]["SLOWFAST"],  # 传递 SLOWFAST 配置
            num_classes=cfg["MODEL"]["NUM_CLASSES"]     # 直接传递类别数
        )
        self.raft = RAFTFlow(cfg["MODEL"]["RAFT"])

        # 光流特征编码器
        self.flow_encoder = nn.Sequential(
            DeformConv3d(6, 64, stride=(1, 2, 2)),  # 输入通道改为6
            nn.MaxPool3d(kernel_size=(1, 3, 3)),
            DeformConv3d(64, 128, stride=(1, 1, 1)),
            DeformConv3d(128, 256, stride=(1, 1, 1))
        )

        # 多模态融合
        self.fusion = DynamicFusion3D(
            slow_dim=1536,  # 直接指定 SlowFast 输出的通道数
            flow_dim=256,
            num_heads=cfg["MODEL"]["FUSION"]["HEADS"]
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, cfg["MODEL"]["NUM_CLASSES"])  # 输入维度需匹配 flow_dim
        )

    def _get_default_config(self, num_classes):
        """生成默认配置（字典结构）"""
        return {
            "MODEL": {
                "NUM_CLASSES": num_classes,
                "SLOWFAST": {"ALPHA": 4, "BETA": 0.125, "DIM": 256},
                "RAFT": {"SMALL": False, "FREEZE": True, "FLOW_SCALE": 1.0},
                "FUSION": {"HEADS": 4}
            }
        }

    def _freeze_raft(self):
        for param in self.raft.parameters():
            param.requires_grad = False
        self.raft.eval()

    def _compute_flow_pyramid(self, video_clip):
        """计算多尺度光流金字塔"""
        B, C, T, H, W = video_clip.size()
        flows = []
        for t in range(T - 1):
            frame1 = video_clip[:, :, t]  # [B,3,H,W]
            frame2 = video_clip[:, :, t + 1]
            flow_full = self.raft(frame1, frame2)  # [B,2,H,W]

            # 调整池化参数以保持空间尺寸一致（使用 stride=1 和 padding）
            flow_1 = F.avg_pool2d(flow_full, kernel_size=2, stride=1, padding=0)  # [B,2,H-1,W-1]
            flow_2 = F.avg_pool2d(flow_full, kernel_size=4, stride=1, padding=0)  # [B,2,H-3,W-3]

            # 插值到原始尺寸（假设 H=224, W=224）
            flow_1 = F.interpolate(flow_1, size=(H, W), mode='bilinear', align_corners=False)
            flow_2 = F.interpolate(flow_2, size=(H, W), mode='bilinear', align_corners=False)

            flows.append(torch.cat([flow_full, flow_1, flow_2], dim=1))  # [B,2+2+2=6,H,W]

        # 补全最后一帧（复制倒数第二帧的光流）
        flows.append(flows[-1].clone())
        return torch.stack(flows, dim=2)  # [B,6,T-1,H,W]

    def forward(self, video_clip):
        # SlowFast路径
        sf_feat = self.slowfast(video_clip)  # [B, 1536, T_slow, H_slow, W_slow]

        # 光流路径
        flow_pyramid = self._compute_flow_pyramid(video_clip)
        flow_feat = self.flow_encoder(flow_pyramid)  # [B, 256, T_flow, H_flow, W_flow]

        # 调整光流特征维度以匹配SlowFast
        target_size = (sf_feat.size(2), sf_feat.size(3), sf_feat.size(4))
        flow_feat = F.interpolate(flow_feat, size=target_size, mode='trilinear', align_corners=False)

        # 确保维度一致
        assert sf_feat.size()[2:] == flow_feat.size()[2:], \
            f"Dimension mismatch after interpolation: {sf_feat.shape} vs {flow_feat.shape}"

        # 跨模态融合
        fused_feat = self.fusion(sf_feat, flow_feat)

        # 分类输出
        pooled = self.classifier[0](fused_feat)
        flattened = self.classifier[1](pooled)
        output = self.classifier[2](flattened)
        return output

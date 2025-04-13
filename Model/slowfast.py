import torch
from torch import nn


class SlowPath(nn.Module):
    def __init__(self, alpha=4):  # 添加 alpha 参数，默认值为 4
        super().__init__()
        self.alpha = alpha  # 初始化 alpha 属性
        # 初始化其他层
        self.conv = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.res_blocks = nn.ModuleList([
            # 示例：添加残差块（根据实际需求调整）
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        ])

    def forward(self, x):
        # x: [B,C,T,H,W]
        x = x[:, :, ::self.alpha, :, :]  # 时间下采样
        x = self.conv(x)
        features = []
        for block in self.res_blocks:  # 遍历每个残差块
            x = block(x)
            features.append(x)  # 收集各阶段特征
        return features  # 返回多阶段特征列表


# 修改 FastPath 类的定义
class FastPath(nn.Module):
    def __init__(self, beta=0.125):
        super().__init__()
        self.beta = beta
        self.conv = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        # 添加多个残差块以生成多阶段特征
        self.res_blocks = nn.ModuleList([
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        ])

    def forward(self, x):
        x = self.conv(x)
        features = []
        for block in self.res_blocks:  # 遍历每个残差块
            x = block(x)
            features.append(x)  # 收集各阶段特征
        return features  # 返回3个特征


class SlowFast(nn.Module):
    """完整SlowFast网络（修复配置层级）"""
    def __init__(self, slowfast_config, num_classes):
        super().__init__()
        self.slow = SlowPath(alpha=slowfast_config["ALPHA"])
        self.fast = FastPath(beta=slowfast_config["BETA"])

        # 横向连接（为每个 slow 特征阶段添加通道对齐卷积）
        self.slow_proj = nn.ModuleList([
            nn.Conv3d(64, 512, kernel_size=1),  # 将 slow 特征的 64 通道映射到 512
            nn.Conv3d(128, 512, kernel_size=1), # 将 slow 特征的 128 通道映射到 512
            nn.Conv3d(256, 512, kernel_size=1)  # 将 slow 特征的 256 通道映射到 512
        ])

        self.lateral_conv = nn.ModuleList([
            nn.Conv3d(64, 512, kernel_size=(5,1,1), stride=(8,1,1), padding=(2,0,0)),
            nn.Conv3d(64, 512, kernel_size=(5,1,1), stride=(8,1,1), padding=(2,0,0)),
            nn.Conv3d(64, 512, kernel_size=(5,1,1), stride=(8,1,1), padding=(2,0,0))
        ])

        # 分类头
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * 3, num_classes)  # 3 个融合特征的总通道数

    def forward(self, x):
        slow_features = self.slow(x)   # 多阶段特征列表 [64, 128, 256]
        fast_features = self.fast(x)   # 多阶段特征列表 [64, 64, 64]

        fused_features = []
        for slow_feat, fast_feat, slow_conv, fast_conv in zip(
            slow_features, fast_features, self.slow_proj, self.lateral_conv
        ):
            # 调整 slow 特征通道数到 512
            adjusted_slow = slow_conv(slow_feat)
            # 调整 fast 特征通道数到 512
            adjusted_fast = fast_conv(fast_feat)
            # 特征融合（相加）
            fused = adjusted_slow + adjusted_fast
            fused_features.append(fused)

        # 拼接所有融合后的特征
        final_feat = torch.cat(fused_features, dim=1)
        return final_feat  # 返回5D特征图 [B, 1536, T, H, W]

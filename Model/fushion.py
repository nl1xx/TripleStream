import torch
from torch import nn


class DynamicFusion3D(nn.Module):
    """
    动态时空特征融合模块
        多头交叉注意力机制
        残差门控连接
    """

    def __init__(self, slow_dim, flow_dim, num_heads=4):
        super().__init__()
        self.slow_proj = nn.Conv3d(slow_dim, flow_dim, 1)
        self.flow_proj = nn.Conv3d(flow_dim, flow_dim, 1)

        # 多头交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=flow_dim,
            num_heads=num_heads,
            kdim=flow_dim,
            vdim=flow_dim
        )

        # 残差门控
        self.gate = nn.Sequential(
            nn.Conv3d(flow_dim * 2, flow_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, slow_feat, flow_feat):
        B, C, T, H, W = flow_feat.size()

        # 确保 slow_feat 和 flow_feat 的 T, H, W 相同
        # assert slow_feat.shape[2:] == (T, H, W), "Feature dimensions mismatch"

        # 动态计算 THW
        THW = T * H * W

        # 投影并调整形状
        slow = self.slow_proj(slow_feat).reshape(B, -1, THW).permute(2, 0, 1)
        flow = self.flow_proj(flow_feat).reshape(B, -1, THW).permute(2, 0, 1)

        # 交叉注意力
        attn_out, _ = self.cross_attn(query=slow, key=flow, value=flow)

        # 使用 reshape 替代 view 以自动处理连续性
        attn_out = attn_out.permute(1, 2, 0).reshape(B, C, T, H, W)  # 关键修改

        # 残差门控
        combined = torch.cat([attn_out, flow_feat], dim=1)
        gate = self.gate(combined)
        return gate * attn_out + (1 - gate) * flow_feat

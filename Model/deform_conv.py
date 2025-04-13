import torch
from torch import nn
from torchvision.ops import deform_conv2d


class DeformConv3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: tuple = (1, 1, 1),  # 修改为元组类型
                 padding: int = 1,
                 dilation: int = 1,
                 deformable_groups: int = 1,
                 norm: bool = True,
                 activation: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride  # 接收元组参数
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

        # 偏移量生成层
        self.offset_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                stride=self.stride,  # 直接使用元组
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=64,
                out_channels=2 * kernel_size ** 2 * deformable_groups,
                kernel_size=(1, kernel_size, kernel_size),
                padding=(0, padding, padding)
            )
        )

        # 主卷积参数
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        # 参数初始化
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

        # 后处理层
        self.norm = nn.BatchNorm3d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        # 生成偏移量 [B, 2*k^2*G, T, H, W]
        offsets = self.offset_conv(x)

        # 逐帧处理
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        outputs = []

        for t in range(T):
            x_t = x[:, t, ...]       # [B, C, H, W]
            offset_t = offsets[:, :, t, ...]  # [B, 2*k^2*G, H, W]

            # 提取空间步幅（忽略时间维度）
            spatial_stride = self.stride[1:]  # 从 (T_stride, H_stride, W_stride) 取后两个

            # 应用可变形卷积（仅传递空间步幅）
            out_t = deform_conv2d(
                input=x_t,
                offset=offset_t,
                weight=self.weight,
                bias=self.bias,
                stride=spatial_stride,  # 使用二维步幅
                padding=self.padding,
                dilation=self.dilation,
                mask=None
            )
            outputs.append(out_t.unsqueeze(1))

        # 重组输出维度
        output = torch.cat(outputs, dim=1)  # [B, T, Cout, H', W']
        output = output.permute(0, 2, 1, 3, 4)  # [B, Cout, T, H', W']

        # 后处理
        return self.act(self.norm(output))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, deformable_groups={self.deformable_groups})"
        )

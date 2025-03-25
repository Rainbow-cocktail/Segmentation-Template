# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentations import get_encoder, EncoderDecoder


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


# 下面是源代码
# class SegFormer(nn.Module):
#     def __init__(self, num_classes=21, encoder_name='b0', pretrained=False):
#         super(SegFormer, self).__init__()
#         self.in_channels = {
#             'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
#             'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
#         }[encoder_name]
#         self.backbone = {
#             'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
#             'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
#         }[encoder_name](pretrained)
#         self.embedding_dim = {
#             'b0': 256, 'b1': 256, 'b2': 768,
#             'b3': 768, 'b4': 768, 'b5': 768,
#         }[encoder_name]
#         self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)
#
#     def forward(self, inputs):
#         H, W = inputs.size(2), inputs.size(3)
#
#         x = self.backbone.forward(inputs)
#         x = self.decode_head.forward(x)
#
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
#         return x

# 下面是我对SegFormer进行的框架适配
class SegFormer(EncoderDecoder):
    def __init__(self, encoder_name='mit_b0', in_channel_nb=3, classes_nb=21):
        # backbone : mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
        encoder, self.in_channels = get_encoder(name=encoder_name,
                                                input_channel=in_channel_nb)
        self.embedding_dim = {
            'mit_b0': 256, 'mit_b1': 256, 'mit_b2': 768,
            'mit_b3': 768, 'mit_b4': 768, 'mit_b5': 768,
        }[encoder_name]
        decoder = SegFormerHead(classes_nb, self.in_channels, self.embedding_dim)

        # 调用 EncoderDecoder 初始化
        super(SegFormer, self).__init__(encoder, decoder)

        self.name = 'SegFormer-{}'.format(encoder_name)

    def forward(self, x):
        """重写forward方法"""
        H, W = x.size(2), x.size(3)  # 记录输入尺寸

        x = self.encoder(x)  # 获取 Transformer backbone 输出的多尺度特征
        x = self.decoder(x)  # 传入 SegFormerHead 解码器

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)  # 还原原尺寸
        return x


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建 SegFormer 模型（使用 b0 版本）
    model = SegFormer(classes_nb=2, encoder_name='mit_b0', in_channel_nb=25).to(device)

    # 创建测试输入图像 (batch_size=4, channels=3, height=512, width=512)
    input_tensor = torch.randn(8, 25, 192, 384).to(device)  # 生成随机图像

    # 运行 SegFormer
    output_mask = model(input_tensor)

    # 打印输出形状
    print("输入图像形状:", input_tensor.shape)
    print("输出分割 mask 形状:", output_mask.shape)

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import os  # 新增：用于创建文件夹
from torchvision.utils import save_image  # 新增：用于保存图片

from .network_utils import (
    ResBlock,
    ConvNormAct,
    # ### MODIFIED ###: 导入我们新增的模块
    Adapter,
    CNNDecoder,
    PredictiveReasoningBlock,
    convert_to_rpm_matrix_v9,
    Classifier
)


class PredRNet(nn.Module):
    def __init__(self, num_filters=32, block_drop=0.0, classifier_drop=0.0,
                 classifier_hidreduce=1.0, in_channels=1, num_classes=8,
                 num_extra_stages=1, reasoning_block=None,  # reasoning_block不再使用
                 num_contexts=8,
                 save_reconstructed_images=True,  # 新增参数：是否保存重建图片
                 save_dir="reconstructed_images",  # 新增参数：保存图片的目录
                 save_frequency=1000):  # 新增参数：保存图片的频率 (例如，每100个batch保存一次)

        super().__init__()

        # --- Part 1: PredRNet Encoder ---
        self.in_planes = in_channels
        channels = [num_filters, num_filters * 2, num_filters * 3, num_filters * 4]
        strides = [2, 2, 2, 2]

        for l in range(len(strides)):
            setattr(self, f"res{l}", self._make_layer(channels[l], strides[l], ResBlock, block_drop))

        self.channel_reducer = nn.Sequential(ConvNormAct(channels[-1], 64, 1, 0, activate=False), nn.MaxPool2d(5))

        # --- Part 2: Adapter ---
        # PredRNet编码器输出为 (32, 5, 5)，RAISE解码器输入为64维
        self.adapter = Adapter(in_channels=num_filters, in_size=5, out_features=64)

        for l in range(3):
            setattr(
                self, "prb"+str(l),
                PredictiveReasoningBlock(64)
            )

        # --- Part 3: RAISE Decoder ---
        # RAISE解码器输出为 (1, 64, 64)
        self.raise_decoder = CNNDecoder(input_dim=64, output_dim=1)

        # 用于在forward过程中存储GT，方便loss计算
        self.ground_truth_panels = None

        # 新增：保存图片相关的参数
        self.save_reconstructed_images = save_reconstructed_images
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        self._save_counter = 0  # 用于跟踪保存频率

        self.classifier = Classifier(
            576, 1,
            norm_layer = nn.BatchNorm1d,
            dropout = classifier_drop,
            hidreduce = classifier_hidreduce
        )

        if self.save_reconstructed_images:
            os.makedirs(self.save_dir, exist_ok=True)  # 创建保存目录

    def _make_layer(self, planes, stride, block, dropout):
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=stride) if stride != 1 else nn.Identity(),
            ConvNormAct(self.in_planes, planes, 1, 0, activate=False)
        )
        stage = block(self.in_planes, planes, downsample, stride=stride, dropout=dropout)
        self.in_planes = planes
        return stage

    def forward(self, images, target):
        """
        重构后的前向传播，用于图像重建
        Args:
            images (Tensor): 整个问题的图像 (B, 16, H, W), e.g., (B, 16, 80, 80)
            target (Tensor): 正确答案的索引 (B,)
        """
        b, _, h, w = images.shape

        # 1. 准备输入：从16个图中选出9个（8上下文+1答案）
        context_panels = images[:, :8]
        correct_answer_panels = images[torch.arange(b), 8 + target].unsqueeze(1)
        panel_input = torch.cat([correct_answer_panels], dim=1)
        self.ground_truth_panels = panel_input

        # 2. PredRNet编码
        # Reshape: (B, 9, H, W) -> (B*9, 1, H, W)
        encoder_input = images.view(b * 16, 1, h, w)

        features = self.res0(encoder_input)
        features = self.res1(features)
        features = self.res2(features)
        features = self.res3(features)
        features = self.channel_reducer(features).squeeze(-1).squeeze(-1)  # Shape: (B*9, 32, 5, 5)
        b,_ = features.shape

        features = features.reshape(b//16, 16, -1)
        context_features = features[:,:8]
        zero_pad = torch.zeros(context_features.size(0), 1, context_features.size(2), device=context_features.device)
        context_features = torch.cat([context_features, zero_pad], dim=1)
        answer_features = features[:,8:]

        for l in range(0, 3):
            context_features = getattr(self, "prb"+str(l))(context_features)
        z = context_features[:,-1].unsqueeze(1)
        err = (answer_features-z)
        x = torch.cat([context_features[:,:8], err], dim=1)
        x = convert_to_rpm_matrix_v9(x, b//16).reshape(b//2, -1)
        out = self.classifier(x).view(b//16, -1)
        # 3. 特征对齐
        # z = self.adapter(features)  # Shape: (B*9, 64)

        # 4. RAISE解码
        reconstructed_images = self.raise_decoder(z)  # Shape: (B*9, 1, 64, 64)

        # ### 新增：保存重建图片逻辑 ###
        if self.save_reconstructed_images and self._save_counter % self.save_frequency == 0:
            # 重建图片通常是 (B*9, 1, H_out, W_out)，需要拆分后保存
            # 为了便于观察，可以把一个batch的9张图（上下文8张+1张正确答案对应的重建图）拼接起来保存
            # 或者只保存正确答案对应的重建图
            # 这里我们尝试保存一个批次中的所有重建图片，每个批次保存一张总图，或者分开保存9张

            # 获取原始的真实图片（用于对比）
            # ground_truth_images = self.ground_truth_panels.view(b * 9, 1, h, w) # (B*9, 1, 80, 80)
            # 注意：如果原始图片大小和重建图片大小不同 (80x80 vs 64x64)，直接拼接会出问题。
            # 这里假设你想保存的是解码器输出的 `reconstructed_images`

            # 示例：保存每个批次的第一组9张重建图片
            # 这里我们假设reconstructed_images是 (B*9, 1, H_out, W_out)
            # 为了方便可视化，我们选择将一个batch中的9张图平铺在一个大图上进行保存
            # 或者可以循环保存每张图片
            for i in range(b//16):  # 遍历每个批次

                img_to_save = reconstructed_images[i]
                # 命名规则：reconstructed_epoch_batch_sample_panel.png
                # 你可能需要在训练循环中传入当前epoch和batch_idx
                # 这里为了演示，我们用一个递增的计数器来命名
                save_path = os.path.join(self.save_dir, f"reconstructed_{self._save_counter}_{i}.png")

                save_image(img_to_save, save_path)
                # print(f"Saved reconstructed image to {save_path}")

        self._save_counter += 1  # 每次forward调用时增加计数器

        return reconstructed_images, out


def predrnet_raven(**kwargs):
    return PredRNet(**kwargs)


def predrnet_analogy(**kwargs):
    raise NotImplementedError("Analogy model not adapted yet.")
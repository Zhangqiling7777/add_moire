#Moiré pattern generation-based image steganography

import torch
import torch.nn as nn

class MoirePatternGenerator(nn.Module):
    def __init__(self):
        super(MoirePatternGenerator, self).__init__()
        
        # 定义五个分支，分别设置不同的上下采样结构
        self.branch1 = self._make_branch(downsample_layers=5, upsample_layers=0)
        self.branch2 = self._make_branch(downsample_layers=4, upsample_layers=1)
        self.branch3 = self._make_branch(downsample_layers=3, upsample_layers=2)
        self.branch4 = self._make_branch(downsample_layers=2, upsample_layers=3)
        self.branch5 = self._make_branch(downsample_layers=1, upsample_layers=4)

    def _make_branch(self, downsample_layers, upsample_layers):
        """创建一个分支的网络结构"""
        layers = []
        
        # 下采样
        for _ in range(downsample_layers):
            layers.append(nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
        
        # 中间卷积块（特征提取）
        layers.extend([
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ])
        
        # 上采样
        for _ in range(upsample_layers):
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            
        # 计算当前分支的输出尺寸
        input_height = 256
        input_width = 256
        current_height = input_height // (2 ** downsample_layers)
        current_width = input_width // (2 ** downsample_layers)
        current_height *= 2 ** upsample_layers
        current_width *= 2 ** upsample_layers
        
        # 如果输出尺寸小于 256x256，则添加上采样层
        while current_height < 256 or current_width < 256:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            current_height *= 2
            current_width *= 2
        
        # 如果输出尺寸大于 256x256，则添加下采样层
        while current_height > 256 or current_width > 256:
            layers.append(nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            current_height //= 2
            current_width //= 2        
        return nn.Sequential(*layers)

    def forward(self, noise, clean_image):
        """
        noise: 随机噪声张量 (batch_size, 3, H, W)
        clean_image: 清晰图像张量 (batch_size, 3, H, W)
        """
        # 每个分支生成一个子摩尔纹图案
        t1 = self.branch1(noise)
        t2 = self.branch2(noise)
        t3 = self.branch3(noise)
        t4 = self.branch4(noise)
        t5 = self.branch5(noise)

        # 合成摩尔纹图案
        moire_patterns = t1 + t2 + t3 + t4 + t5

        # 将摩尔纹叠加到清晰图像上
        synthesized_image = moire_patterns + clean_image

        return synthesized_image

# 测试生成器
if __name__ == "__main__":
    generator = MoirePatternGenerator()
    noise = torch.randn(1, 3, 256, 256)  # 随机噪声输入
    clean_image = torch.randn(1, 3, 256, 256)  # 清晰图像输入
    output = generator(noise, clean_image)
    print("输出图像尺寸:", output.shape)

import torch
import torch.nn as nn
from torchvision import models


# 定义InceptionNetV3模型
class InceptionNetV3(nn.Module):
    def __init__(self, num_classes=3):
        super(InceptionNetV3, self).__init__()
        inception = models.inception_v3(pretrained=True)  # 加载预训练的InceptionNetV3模型
        # 冻结所有层
        for param in inception.parameters():
            param.requires_grad = False
        # 从InceptionNetV3模型中提取特征部分（卷积层）
        self.features = inception.features
        # 添加自定义的分类层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

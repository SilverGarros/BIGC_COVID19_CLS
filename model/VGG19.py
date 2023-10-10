import torch
import torch.nn as nn
from torchvision import models


# 定义VGG19模型
class VGG19(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG19, self).__init__()
        vgg19 = models.vgg19(pretrained=False)  # 加载预训练的VGG19模型
        # 冻结所有层
        for param in vgg19.parameters():
            param.requires_grad = False
        # 从VGG19模型中提取特征部分（卷积层）
        self.features = vgg19.features
        # 添加自定义的分类层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

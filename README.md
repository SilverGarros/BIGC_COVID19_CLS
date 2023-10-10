# BIGC_COVID19_CLS
## 北京印刷学院人工智能大赛深度学习组作品_基于深度学习的新冠肺炎检测系统

本项目设计出多种基于深度学习神经网络模型的新冠肺炎影像特征识别系统以及用于用户进行检测的Web程序。该系统通过对比主流卷积神经网络(CNN)模型，VGG16与VGG19模型以及InceptionNetV3深度学习模型，并通过实验对比算法识别的精确度及进行相应模型参数调整优化，验证多种模型在识别新冠肺炎CT图像中的有效性，导出训练后的模型权重，使用Flask框架构建了一个简单的WebApp，允许用户通过远程访问上传CT图片选择不同的模型进行肺部健康状况的检测。

权重文件见云盘
## 仓库结构
![仓库结构图](https://github.com/SilverGarros/BIGC_COVID19_CLS/assets/93498846/3a832bb0-248a-4fd9-9bfd-16e3baa799d2)
## 训练过程记录
Accuracy变化情况
![训练过程记录Accuracy变化情况](https://github.com/SilverGarros/BIGC_COVID19_CLS/assets/93498846/4b771d1a-196e-470e-9e89-abf9e8a3a405)
Loss变化情况
![训练过程记录Loss变化情况](https://github.com/SilverGarros/BIGC_COVID19_CLS/assets/93498846/05c2c35b-1467-4afd-aece-6c101665d939)
## Web展示图
![Web展示图](https://github.com/SilverGarros/BIGC_COVID19_CLS/assets/93498846/3dba3e85-5829-4351-9d52-24b04a848e96)
![Web操作展示图1](https://github.com/SilverGarros/BIGC_COVID19_CLS/assets/93498846/9c67c08b-ddf9-4ea9-8c4d-8e5a09f98971)

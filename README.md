# BIGC_COVID19_CLS
北京印刷学院人工智能大赛深度学习组作品_基于深度学习的新冠肺炎检测系统

本项目设计出多种基于深度学习神经网络模型的新冠肺炎影像特征识别系统以及用于用户进行检测的Web程序。该系统通过对比主流卷积神经网络(CNN)模型，VGG16与VGG19模型以及InceptionNetV3深度学习模型，并通过实验对比算法识别的精确度及进行相应模型参数调整优化，验证多种模型在识别新冠肺炎CT图像中的有效性，导出训练后的模型权重，使用Flask框架构建了一个简单的WebApp，允许用户通过远程访问上传CT图片选择不同的模型进行肺部健康状况的检测。

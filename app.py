import torchvision
from flask import Flask, render_template, request, session
from PIL import Image
import torch
import torchvision.transforms as transforms
from model.CNN import CNNModel  # 导入你的模型定义
from model.VGG16 import VGG16
from model.VGG19 import VGG19
import os
import time
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'BIGC_204010218'
import torchvision.models as models

# 初始化模型映射
available_models = {
    'CNN': {
        'model': CNNModel(),
        'weight_path': './weights/CNN/CNN_best_model_weights.pth',
    },
    'VGG16': {
        'model': VGG16(),
        'weight_path': './weights/VGG16/VGG16_best_model_weights.pth',
    },
    'VGG19': {
        'model': VGG19(),
        'weight_path': './weights/VGG19/VGG19_best_model_weights.pth',
    },
    'InceptionNetV3': {
        'model': models.inception_v3(pretrained=False),  # 使用torchvision中的InceptionV3模型
        'weight_path': './weights/InceptionNetV3/inceptionnet_best_model_weights.pth',
    },
}

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 存储历史检测结果和上传图像的目录
history_results = []
upload_dir = './static/uploads'  # 存储上传图像的目录
os.makedirs(upload_dir, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def classify_image():
    uploaded_image_url = None
    result = None
    selected_model = None
    uploaded_image_dir_name = None
    if request.method == 'POST':
        uploaded_file = request.files['file']
        selected_model = request.form.get('model')  # 获取用户选择的模型

        if uploaded_file.filename != '' and selected_model in available_models:
            print(uploaded_file)
            # 生成上传文件的唯一文件名
            timestamp = time.mktime(datetime.now().timetuple())
            image_filename = os.path.join(upload_dir, f'{int(timestamp)}.jpg')

            # 保存上传的文件
            uploaded_file.save(image_filename)
            uploaded_image_dir = image_filename  # 存储上传的图像路径
            print("保存文件" + str(uploaded_image_dir))

            # 获取选定模型的模型对象和权重路径
            selected_model_info = available_models[selected_model]
            model = selected_model_info['model']
            weight_path = selected_model_info['weight_path']
            print(weight_path)

            # 使用选择的模型进行分类
            prediction = get_prediction(image_filename, model, weight_path)

            # 根据选择的模型生成对应的结果
            if prediction == 0:
                result = 'COVID-19'
            elif prediction == 1:
                result = 'NORMAL'
            elif prediction == 2:
                result = 'Viral Pneumonia'
            print(history_results)
            # 将结果添加到历史检测结果列表中
            uploaded_image_dir = uploaded_image_dir.replace("templates/", "")
            # 更新uploaded_image_url为图像的URL
            uploaded_image_url = f"{image_filename}"  # 图像的URL
            history_results.append((uploaded_file.filename, selected_model, result,
                                    str(datetime.fromtimestamp(timestamp)), uploaded_image_url))
            print(uploaded_file.filename)
            # 更新uploaded_image_url为图像的URL
            uploaded_image_url = f"{image_filename}"  # 图像的URL
            # 将历史结果保存在session中
            session['history_results'] = history_results
            history_results.reverse()
            print(history_results)
            uploaded_image_dir_name = uploaded_file.filename

    return render_template('index.html', result=result,
                           uploaded_image_url=uploaded_image_url,
                           history_results=history_results, selected_model=selected_model,
                           available_models=available_models, uploaded_image_dir=uploaded_image_dir_name)


# 在 get_prediction 函数中，不需要再次创建模型实例
def get_prediction(image_path, model, weight_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')  # 如果不是RGB，转换为RGB
    image = transform(image).unsqueeze(0)

    # 加载模型的预训练权重
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()  # 设置模型为评估模式

    # 使用模型进行分类
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()


if __name__ == '__main__':
    app.run(debug=True)

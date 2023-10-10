import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

# 指定预训练权重文件路径
pretrained_weights_path = './pre-trained weights/VGG16.pth'

# 初始化TensorBoard的SummaryWriter
writer = SummaryWriter(log_dir='./logs')

# 定义数据增强和加载数据
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('../dataset_jpg/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder('../dataset_jpg/test', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 使用自定义预训练权重路径或下载到pretrained_weights_path
if not os.path.exists(pretrained_weights_path):
    # 下载VGG16预训练权重
    print("Downloading VGG16 pretrained weights...")
    model_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
    model_zoo.load_url(model_url, model_dir='./pre-trained weights/vgg16.pth')

# 使用预训练的VGG16模型
model = models.vgg16(pretrained=False)  # 不使用内置的预训练权重
model.load_state_dict(torch.load(pretrained_weights_path))  # 加载自定义预训练权重

# 修改最后一个全连接层，以适应分类任务（3个类别）
model.classifier[6] = nn.Linear(4096, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# 打印模型摘要
# summary(model, (3, 224, 224), device=device)
print(model)

criterion = nn.CrossEntropyLoss()

# 初始化最佳精度和对应的模型权重
best_accuracy = 0.0
best_model_weights = None
weight_save_path = '../weights/VGG16/'
os.makedirs(weight_save_path, exist_ok=True)

# 开始训练
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{total_batches}], Loss: {loss.item()}")

    writer.add_scalar('Train Loss', running_loss / total_batches, epoch)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    writer.add_scalar('Test Accuracy', accuracy, epoch)
    if epoch/5 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / total_batches}, Test Accuracy: {accuracy}%")

    scheduler.step()

    torch.save(model.state_dict(), weight_save_path + f'model_weights_epoch_{epoch + 1}.pth')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_weights = model.state_dict()

torch.save(best_model_weights, weight_save_path + 'VGG16_best_model_weights.pth')

writer.close()

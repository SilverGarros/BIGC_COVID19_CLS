import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

# 初始化TensorBoard的SummaryWriter
writer = SummaryWriter()
# 记录超参数（例如学习率）
writer.add_hparams({'lr': 0.001, 'batch_size': 32}, {'hparam/accuracy': 0.95, 'hparam/loss': 0.1})

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

# 使用预训练的vgg19模型
model = models.vgg19(pretrained=True)

# 冻结模型的所有层（可选）
for param in model.parameters():
    param.requires_grad = False

# 修改最后一个全连接层，以适应你的分类任务（3个类别）
model.classifier[6] = nn.Linear(4096, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


criterion = nn.CrossEntropyLoss()

# 初始化最佳精度和对应的模型权重
best_accuracy = 0.0
best_model_weights = None
weight_save_path = '../weights/vgg19/'
os.makedirs(weight_save_path, exist_ok=True)

# 开始训练
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)  # 获取批次总数
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 实时显示损失
        print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{total_batches}], Loss: {loss.item()}")

    # 使用TensorBoard记录训练损失
    writer.add_scalar('Train Loss', running_loss / total_batches, epoch)

    # 在测试集上评估模型
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

    # 使用TensorBoard记录测试精度
    writer.add_scalar('Test Accuracy', accuracy, epoch)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / total_batches}, Test Accuracy: {accuracy}%")

    # 更新学习率
    scheduler.step()

    # 保存每轮训练的权重
    torch.save(model.state_dict(), weight_save_path + f'model_weights_epoch_{epoch + 1}.pth')

    # 更新最佳模型权重
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_weights = model.state_dict()

# 保存最佳权重
torch.save(best_model_weights, weight_save_path + 'vgg19_best_model_weights.pth')

# 关闭TensorBoard的SummaryWriter
writer.close()

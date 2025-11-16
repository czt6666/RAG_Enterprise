import torch
from torch import nn  # 用于构建模型和损失函数
from torch.utils.data import DataLoader  # 用于构建数据加载器
from torchvision import datasets, transforms  # 用于读入和变换数据集

# 优先用英伟达GPU，没有的话用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# 定义数据变换：将图片从[0,255]像素值范围转换为[0,1]，并将通道顺序从HWC转为CHW
transform = transforms.ToTensor()
# 下载并加载和变换训练集
train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
# 批次大小为128，每个epoch打乱样本顺序
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# 下载并加载和变换测试集
test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 定义模型类
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # 输入CIFAR10图像为32x32大小的彩色图像，3通道
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),  # 该卷积层不改变特征图大小
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层，特征图长宽都变为原来一半
        )
        self.flatten = nn.Flatten()  # 把特征图拉直
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),  # 全连接层，输入是拉直的特征图
            nn.ReLU(),
            nn.Linear(256, 10)  # 输出层，输出神经元数量需要和预测类别数一致
        )

    def forward(self, x):  # 定义前向传播计算
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


model = ConvNet().to(device)  # 将模型实例化并放在计算设备上
loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
# loss_fn = nn.BCEWithLogitsLoss() # 如果是二分类使用二元交叉熵损失函数
# loss_fn = nn.MSELoss() # 如果是回归使用均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 使用随机梯度下降法优化参数，学习率为0.001


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 样本量
    num_batches = len(dataloader)  # batch数量
    cum_samples = 0  # 已经访问过的样本量
    model.train()  # 模型切换成训练模式
    for batch, (X, y) in enumerate(dataloader):  # 取出每个batch的数据
        X, y = X.to(device), y.to(device)  # 将数据放在计算设备上
        pred = model(X)  # 使用模型进行前向传播
        loss = loss_fn(pred, y)  # 计算损失

        optimizer.zero_grad()  # 将前一个batch计算出的梯度清空（置零）
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 梯度下降

        cum_samples += X.shape[0]  # 累积已经访问过的样本量
        if (batch + 1) % 100 == 0 or (batch + 1) == num_batches:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{cum_samples: >5d} / {size: >5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # 模型切换为评估（测试、推断）模式
    test_loss, correct = 0.0, 0
    with torch.no_grad():  # 无需构建计算图来计算偏导数，节省显存和时间
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * X.shape[0]
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
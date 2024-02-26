#使用前需将文件名修改为main
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 文件路径
test_file_path = r'F:\python-learn\test（mnist）.csv'
train_file_path = r'F:\python-learn\train（mnist）.csv'
# 读取CSV文件
test_data = pd.read_csv(test_file_path)
train_data = pd.read_csv(train_file_path)
# 划分训练集和验证集
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
# 计算数据集大小
train_data_size = len(train_data)
val_data_size = len(val_data)
test_data_size = len(test_data)
print(f'测试集长度为{test_data_size}')
print(f'训练集长度为{train_data_size}')
print(f'验证集长度为{val_data_size}')
# 转换为张量
train_data_tensor = torch.tensor(train_data.values, dtype=torch.float32)
val_data_tensor = torch.tensor(val_data.values, dtype=torch.float32)
test_data_tensor = torch.tensor(test_data.values, dtype=torch.float32)
# 划分特征和标签
train_features, train_labels = train_data_tensor[:, 1:], train_data_tensor[:, 0]
val_features, val_labels = val_data_tensor[:, 1:], val_data_tensor[:, 0]
test_features = test_data_tensor[:, 0:]
# 调整特征形状
train_features = train_features.view(-1, 1, 28, 28)
val_features = val_features.view(-1, 1, 28, 28)
test_features = test_features.view(-1, 1, 28, 28)
# 使用TensorDataset创建DataLoader
train_dataset = TensorDataset(train_features, train_labels.long())  # 假设标签是整数类型
val_dataset = TensorDataset(val_features, val_labels.long())
# 为测试数据创建一个没有标签的Dataset
test_dataset = TensorDataset(test_features)
# 创建DataLoader
train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=0),#一个灰度图像的卷积层，输入通道数为1，输出通道数为6，卷积核大小为5x5，步幅为1，没有填充（padding=0）
            nn.MaxPool2d(kernel_size=(2,2)),#池化核的大小是(2, 2)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0), # 一个灰度图像的卷积层，输入通道数为6，输出通道数为16，卷积核大小为5x5，步幅为1，没有填充（padding=0）
            nn.MaxPool2d(kernel_size=(2, 2)),  # 池化核的大小是(2, 2)
            nn.Flatten(),#展平成一个一维向量，以便连接到全连接层或进行其他操作
            nn.Linear(in_features=16*4*4,out_features=120),#这个线性层的输入特征数量是 16 * 4 * 4，这通常是因为前面经过了一个卷积层，输出了一个大小为120的特征图
            nn.Linear(in_features=120, out_features=84),# 这个线性层的输入特征数量是 120，这通常是因为前面经过了一个卷积层，输出了一个大小为 84 的特征图
            nn.Linear(in_features=84, out_features=10),# 这个线性层的输入特征数量是 84，这通常是因为前面经过了一个卷积层，输出了一个大小为 10 的特征图
        )
    def forward(self,x):#这是一个 PyTorch 模型的前向传播方法（forward 方法）,该方法描述了输入如何通过网络层传递以生成输出。
        x = self.model(x)
        return x
#实例化网络
mynet = MyNet()
mynet=mynet.to(device)#利用GPU加速网络
#print(mynet)
loss_fn = nn.CrossEntropyLoss()#创建一个用于计算交叉熵损失的 PyTorch 损失函数对象
loss_fn=loss_fn.to(device)#利用GPU加速损失函数
learning_rate = 1e-3
optim = torch.optim.SGD(mynet.parameters(),learning_rate)#创建了一个用于优化你的神经网络模型 mynet 的随机梯度下降（SGD）优化器，并设置了学习率0.001
train_step = 0
epoch = 20#迭代次数
if __name__=='__main__':
    for i in range(epoch):
        print(f'----------第{i+1}轮训练----------')
        mynet.train()
        for data in train_data_loader:
            imgs,targets = data#数据拆包为图像（imgs）和目标标签（targets）
            #利用GPU加速imgs和targets
            imgs=imgs.to(device)
            targets=targets.to(device)
            #print(imgs.shape)
            outputs = mynet(imgs)
            #print(outputs.shape)
            loss = loss_fn(outputs,targets)#使用定义的损失函数计算模型输出与真实标签之间的损失
            optim.zero_grad()#清零之前的梯度，以避免梯度累积
            loss.backward()#反向传播，计算梯度。
            optim.step()#根据计算的梯度更新模型的参数（权重）
            train_step += 1
            if train_step%100==0:
                print(f"第{train_step}次训练,loss={loss.item()}")
        mynet.eval()
        accuracy=0
        total_accuracy=0
        with torch.no_grad():
            for data in val_data_loader:
                imgs, targets = data  # 数据拆包为图像（imgs）和目标标签（targets）
                # 利用GPU加速imgs和targets
                imgs = imgs.to(device)
                targets = targets.to(device)
                # print(imgs.shape)
                outputs = mynet(imgs)
                # print(outputs.shape)
                accuracy=(outputs.argmax(axis=1) == targets).sum()#计算了正确预测的数量
                total_accuracy+=accuracy
            print(f"{i+1}轮训练结束，准确率为{total_accuracy/val_data_size}")
            #将训练好的模型保存到文件，以便稍后在其他地方加载和使用
            #torch.save(mynet,f'MNIST_{i}_acc_{total_accuracy/val_data_size}.pth')
# 用于存储测试集的预测结果
predicted_results = []
with torch.no_grad():#该上下文环境中禁用梯度计算
    for data in test_data_loader:
        imgs = torch.stack(data)
        # 使用squeeze方法删除第一个维度为1的维度
        imgs = imgs.squeeze(dim=0)
        imgs = imgs.to(device)
        #print(imgs.shape)
        outputs = mynet(imgs)
        #print(outputs.shape)
        # 使用softmax函数获取概率并得到预测的类别
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_classes = torch.max(probabilities, 1)
        # 将预测的类别添加到结果列表中
        for pred_class in predicted_classes:
            predicted_results.append(pred_class.item())
# 打印预测结果
print("预测结果:", predicted_results)
# 将结果保存到Excel文件
df = pd.DataFrame({'预测结果': predicted_results})
df.to_excel('predictions.xlsx', index=False)
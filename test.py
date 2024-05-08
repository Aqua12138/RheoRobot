import torch
import torch.nn as nn

# 定义神经网络结构
class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(10, 20)  # 输入层到隐藏层1，假设输入特征维度是10
        self.layer2 = nn.Linear(20, 30)  # 隐藏层1到隐藏层2
        self.output_layer = nn.Linear(30, 5)  # 隐藏层2到输出层，假设有5个类别

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

# 创建模型实例
model = SimpleNeuralNetwork()

# 假设输入数据的batch size为4，每个样本特征维度为10
input_data = torch.zeros(4, 10)  # 随机生成一些输入数据

# 通过网络计算输出结果
output_data = model(input_data)

# 打印输入和输出的维度
print("Input shape:", input_data)  # 应显示torch.Size([4, 10])
print("Output shape:", output_data)  # 应显示torch.Size([4, 5])

import torch
import random
import numpy as np
from d2l import torch as d2l

torch.set_printoptions(edgeitems=2, precision=6, linewidth=100, sci_mode=False)


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个Pytorch数据迭代器"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
print(f'true_w {true_w}, true_b {true_b}')

batch_size = 10
data_iter = load_array((features, labels), batch_size)
# print(next(iter(data_iter)))

# 定义一个包含单个线性层的简单神经网络net，输入特征维度为2，输出维度为1
net = torch.nn.Sequential(torch.nn.Linear(2, 1))
# 用正态分布初始化线性层的权重
net[0].weight.data.normal_(0, 0.01)
# 偏置初始化为0
net[0].bias.data.fill_(0)

# 损失函数使用均方误差。默认情况下，它返回所有样本损失的平均值
loss = torch.nn.MSELoss()
# 小批量随机梯度下降算法是一种优化神经网络的标准工具， PyTorch在optim模块中实现了该算法的许多变种。
# 当我们(实例化一个SGD实例)时，我们要指定优化的参数 （可通过net.parameters()从我们的模型中获得）
# 以及优化算法所需的超参数字典。 小批量随机梯度下降只需要设置lr值，这里设置为0.03。
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        # 清零优化器中参数的梯度
        trainer.zero_grad()
        # 进行反向传播计算梯
        l.backward()
        # 根据计算得到的梯度更新模型参数
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.item():f}, w {net[0].weight.data}, b {net[0].bias.data}')

import torch
import numpy as np
import random
from d2l import torch as d2l
from torch import mean

torch.set_printoptions(edgeitems=2, precision=6, linewidth=75, sci_mode=False)


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + noise"""
    X = torch.normal(0, 1, (num_examples, len(w))) #使用均值为1，标准差为1的高斯分布
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) #添加均值为0，标准差为0.01的噪声
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
# print(features.shape, labels.shape)

# 显示散点图
# d2l.set_figsize()
# d2l.plt.scatter(features[:,0].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()

def data_iter(batch_size, features, labels):
    """
    接收批量大小、特征矩阵和标签向量作为输入，生成大小为`batch_size`的小批量。
    每个小批量包含一组特征和标签。
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机读取样本
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)

# 初始化模型参数
# 通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，并将偏置初始化为0。
w = torch.normal(mean=0, std=0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """损失函数：均方误差"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, learn_rate, batch_size):
    """
    实现小批量随机梯度下降更新。
    该函数接受模型参数集合、学习速率和批量大小作为输入。每
    一步更新的大小由学习速率`lr`决定。
    因为我们计算的损失是一个批量样本的总和，所以我们用批量大小（`batch_size`）
    来规范化步长，这样步长大小就不会取决于我们对批量大小的选择。
    """
    with torch.no_grad():
        for param in params:
            param -= learn_rate * param.grad / batch_size
            param.grad.zero_()

# 开始训练模型
learn_rate = 0.03
num_epochs = 3
net_func = linreg
loss_func = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        loss = loss_func(net_func(X, w, b), y) # X和y的小批量损失
        # 因为loss形状是(batch_size,1)，而不是一个标量。loss中的所有元素被加到一起，并以此计算关于[w,b]的梯度
        loss.sum().backward()
        sgd([w, b], learn_rate, batch_size) # 使用参数的梯度更新参数
    with torch.no_grad():
        train_loss = loss_func(net_func(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')
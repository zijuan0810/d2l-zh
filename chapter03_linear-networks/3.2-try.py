import torch
import random
import numpy as np
from d2l import torch as d2l

torch.set_printoptions(edgeitems=2, precision=6, linewidth=100, sci_mode=False)


def synthetic_data(w, b, num_examples):
    """生成模拟数据：y = Xw + b + noise"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # 使用均值为1，标准差为1的高斯分布
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 添加均值为0，标准差为0.01的噪声
    return X, y.reshape((-1, 1))


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
    random.shuffle(indices)  # 将索引值随机打乱，随机读取样本
    for i in range(0, num_examples, batch_size):
        end = min(i + batch_size, num_examples)  # 最大值不能超过样本数量
        batch_indices = torch.tensor(indices[i:end])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """线性回归模型：y = Xw + b"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y_true):
    """损失函数：均方误差"""
    return (y_hat - y_true.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, learn_rate, batch_size):
    """
    实现小批量随机梯度下降更新。
    该函数接受模型参数集合、学习速率和批量大小作为输入。每一步更新的大小由学习速率`lr`决定。
    因为我们计算的损失是一个批量样本的总和，所以我们用批量大小 batch_size 来规范化步长，
    这样步长大小就不会取决于我们对批量大小的选择。
    """
    with torch.no_grad():
        for param in params:
            param -= learn_rate * param.grad / batch_size
            param.grad.zero_()


# 生成模拟数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 初始化模型参数
# 通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，并将偏置初始化为0。
# 使用全零的w也不影响模型的结果 torch.zeros((2, 1), requires_grad=True)
w = torch.normal(mean=0, std=0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 开始训练模型，设置超参数
learn_rate = 0.01
num_epochs = 10
batch_size = 10
net_func = linreg
loss_func = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        loss = loss_func(net_func(X, w, b), y)  # X和y的小批量损失
        # 因为loss形状是(batch_size,1)，而不是一个标量。loss中的所有元素被加到一起，并以此计算关于[w,b]的梯度
        # 通过调用 sum() 函数是为了计算损失的总和。这是因为 loss 的形状是 (batch_size, 1)， 它包含了一个小批量
        # 样本的损失值。调用 sum() 函数可以将所有损失值相加，得到一个标量值，这样就可以计算出标量对应的梯度。
        #
        # 在 PyTorch 中，反向传播过程中要求对标量值进行求导，因此需要将损失值求和得到一个标量。通过调用 sum() 函数，
        # 我们可以将 loss 中所有元素相加得到总和，然后进行反向传播以计算参数的梯度。这个过程将使用链式法则来计算关于
        # 模型参数的梯度，并将梯度累积在参数的 .grad 属性中。
        #
        # 请注意，由于每次迭代中都会计算一个小批量样本的损失总和，并在后续的参数更新中使用该梯度，
        # 所以在进行下一次迭代之前，需要使用 param.grad.zero_() 将参数的梯度清零，以避免梯度的累积影响后续的计算。
        #
        # 总之，loss.sum().backward() 的目的是计算损失的总和，并进行反向传播以计算模型参数的梯度。
        loss.sum().backward()
        sgd([w, b], learn_rate, batch_size)  # 使用参数的梯度更新参数

    # 仅仅是为了打印每个epoch完成后的损失值
    with torch.no_grad():
        # 利用当前epoch中计算出来的w和b来计算模型在整个训练集上的损失
        train_loss = loss_func(net_func(features, w, b), labels)
        print(f'epoch {epoch + 1}, w {w}, b {b} loss {float(train_loss.mean()):f}')

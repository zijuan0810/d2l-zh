import torch
from IPython import display
import torchvision
from pandas.conftest import keep
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
torch.set_printoptions(edgeitems=2, precision=6, linewidth=120, sci_mode=False)

"""================================================================="""
# 定义模型超参数
learn_rate = 0.1
batch_size = 256

# 初始化模型参数
# 原始数据集中的每个样本都是$28 \times 28$的图像。
# 本节将展平每个图像，把它们看作长度为784的向量。
# 因为我们的数据集有10个类别，所以网络输出维度为10
# 权重将构成一个784x10的矩阵，
# 偏置将构成一个1x10的行向量。
# 与线性回归一样，我们将使用正态分布初始化我们的权重W，偏置初始化为0
num_inputs = 784  # 28 * 28
num_outputs = 10  # 10个类别
W = torch.normal(0, std=0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
"""================================================================="""


class Accumulator:
    """用于对多个变量进行累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def softmax(X, keepdim=True):
    """定义softmax操作"""
    X_exp = torch.exp(X)
    # print(f'X_exp shape: {X_exp.shape}')
    partition = X_exp.sum(1, keepdim=keepdim)  # 计算每行的总和
    # print(f'partition: {partition.shape} {partition}')
    return X_exp / partition  # 这里引用了torch的广播机制


# 定义模型
def net(X):
    """定义了输入如何通过网络映射到输出"""
    v = X.reshape((-1, W.shape[0]))  # 将每张原始图像展平为向量 [784,10] => [256, 784]
    return softmax(torch.matmul(v, W) + b)  # v[256,784] * W[784,10]


def cross_entropy(y_hat, y):
    """定义交叉熵损失函数"""
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    # y_hat是矩阵，那么第二个维度存储每个类的预测分数
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # 由于等式运算符“`==`”对数据类型很敏感，
    # 因此我们将`y_hat`的数据类型转换为与`y`的数据类型一致。
    # 结果是一个包含0（错）和1（对）的张量。
    cmp = y_hat.type(y.dtype) == y
    # 求和会得到正确预测的数量
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    """我们定义一个函数来训练一个迭代周期。
    请注意，`updater`是更新模型参数的常用函数，它接受批量大小作为参数。
    它可以是`d2l.sgd`函数，也可以是框架的内置优化函数。
    """
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数量
    metric = Accumulator(3)
    for X, y in train_iter:
        # X.shape:[256, 1, 28, 28] y.shape:[256]
        y_hat = net(X)  # [256,10]
        l = loss(y_hat, y)  # [256]
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        # y.numel()获取tensor中一共包含多少个元素，即样本数量
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    train_metrics = None
    test_acc = 0.0
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        # 输出结果
        print(f'epoch {epoch + 1}, train loss {train_metrics[0] + test_acc:.4f}  '
              f'train acc {train_metrics[1]:.4f}, test_acc {test_acc:.4f}')
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 1 >= test_acc > 0.7, test_acc


def updater(batch_size):
    return d2l.sgd([W, b], learn_rate, batch_size)


# 现在训练已经完成，我们的模型已经准备好对图像进行分类预测
# 给定一系列图像，我们将比较它们的实际标签（文本输出的第一行）和模型预测（文本输出的第二行）
def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    X, y = next(iter(test_iter))  # 只迭代一次
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


if __name__ == '__main__':
    # 加载数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 训练模型10个迭代周期
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    predict_ch3(net, test_iter)
    d2l.plt.show()

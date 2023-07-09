import torch
from IPython import display
import torchvision
from pandas.conftest import keep
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
torch.set_printoptions(edgeitems=2, precision=6, linewidth=120, sci_mode=False)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

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


# 定义softmax操作
# X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# print(X.shape) # torch.Size([2, 3])
# print(X.sum(0, keepdim=True)) # tensor([[5., 7., 9.]]) 即去掉第一个轴，[1, 3]
# print(X.sum(1, keepdim=True)) # tensor([[ 6.], [15.]]) 即去掉第二个轴，[2, 1]


def softmax(X, keepdim=True):
    X_exp = torch.exp(X)
    # print(f'X_exp shape: {X_exp.shape}')
    partition = X_exp.sum(1, keepdim=keepdim)
    # print(f'partition: {partition.shape} {partition}')
    return X_exp / partition  # 这里引用了torch的广播机制


# 测试验证代码
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(1))  # 计算每行的总和，等于1，说明所有类别的总概率为1


# 定义模型
def net(X):
    """定义了输入如何通过网络映射到输出"""
    v = X.reshape((-1, W.shape[0]))  # 将每张原始图像展平为向量
    return softmax(torch.matmul(v, W) + b)


# 下面，我们[**创建一个数据样本`y_hat`，其中包含2个样本在3个类别的预测概率，
# 以及它们对应的标签`y`。**]
# 有了`y`，我们知道在第一个样本中，第一类是正确的预测；
# 而在第二个样本中，第三类是正确的预测。
# 然后(**使用`y`作为`y_hat`中概率的索引**)，
# 我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率。
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])  # shape [2, 3]
print(y_hat[[0, 1], y])  # 这里的[0,1]表示行数


def cross_entropy(y_hat, y):
    """定义交叉熵损失函数"""
    return -torch.log(y_hat[range(len(y_hat)), y])


print(cross_entropy(y_hat, y))


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


print(accuracy(y_hat, y) / len(y))


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


class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

        d2l.plt.show()


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 由于我们使用随机权重初始化`net`模型，
# 因此该模型的精度应接近于随机猜测。
# 例如在有10个类别情况下的精度为0.1。
print(evaluate_accuracy(net, test_iter))


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
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    train_metrics = None
    test_acc = 0.0
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 1 >= test_acc > 0.7, test_acc


lr = 0.1


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


# 现在，我们[**训练模型10个迭代周期**]。
# 请注意，迭代周期（`num_epochs`）和学习率（`lr`）都是可调节的超参数。
# 通过更改它们的值，我们可以提高模型的分类精度。
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


# 现在训练已经完成，我们的模型已经准备好[**对图像进行分类预测**]。
# 给定一系列图像，我们将比较它们的实际标签（文本输出的第一行）和模型预测（文本输出的第二行）。
def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    # for X, y in test_iter:
    #     break
    X, y = next(iter(test_iter))
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


predict_ch3(net, test_iter)
d2l.plt.show()

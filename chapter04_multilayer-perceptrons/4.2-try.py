import torch
from torch import nn
from d2l import torch as d2l

"""
定义一个具有单隐藏层的多层感知机（MLP）模型，并使用 Fashion-MNIST 数据集进行训练和测试
"""

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784  # 输入层的输入数目，等于图像的大小28x28
num_outputs = 10  # 输出层的输出数目，等于类别的数目，这里为10
num_layer1 = 256  # 隐藏层layer1的神经元数目
num_layer2 = 128  # 隐藏层layer2的神经元数目
num_layer3 = 64  # 隐藏层layer3的神经元数目

W1 = nn.Parameter(torch.randn(num_inputs, num_layer1, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_layer1, requires_grad=True))

W2 = nn.Parameter(torch.randn(num_layer1, num_layer2, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_layer2, requires_grad=True))

W3 = nn.Parameter(torch.randn(num_layer2, num_layer3, requires_grad=True) * 0.01)
b3 = nn.Parameter(torch.zeros(num_layer3, requires_grad=True))

W4 = nn.Parameter(torch.randn(num_layer3, num_outputs, requires_grad=True) * 0.01)
b4 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

# W2 = nn.Parameter(torch.randn(num_layer1, num_outputs, requires_grad=True) * 0.01)
# b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2, W3, b3, W4, b4]


def relu(X):
    """激活函数"""
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    """因为我们忽略了空间结构，
    所以我们使用`reshape`将每个二维图像转换为一个长度为`num_inputs`的向量。
    只需几行代码就可以(**实现我们的模型**)
    """
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    return H @ W2 + b2


if __name__ == '__main__':
    loss = nn.CrossEntropyLoss(reduction='none')
    num_epochs, learn_rate = 10, 0.1
    updater = torch.optim.SGD(params, lr=learn_rate)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

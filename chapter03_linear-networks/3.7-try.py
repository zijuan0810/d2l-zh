import torch
from d2l import torch as d2l

d2l.use_svg_display()
torch.set_printoptions(edgeitems=2, precision=6, linewidth=120, sci_mode=False)

# 初始化模型参数
# softmax回归的输出层是一个全连接层
# 我们只需在`Sequential`中添加一个带有10个输出的全连接层
# 同样，在这里`Sequential`并不是必要的， 但它是实现深度模型的基础
# 我们仍然以均值0和标准差0.01随机初始化权重。
net = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 10),
    torch.nn.Softmax(dim=1))
# et = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)
        torch.nn.init.constant_(m.bias, 0)


net.apply(init_weights)

# 定义损失函数和优化器
loss = torch.nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型
num_epochs = 10
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3_noani(net, train_iter, test_iter, loss, num_epochs, trainer)

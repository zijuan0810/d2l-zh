import torch
from IPython.core.pylabtools import figsize

from d2l import torch as d2l

d2l.use_svg_display()
torch.set_printoptions(edgeitems=2, precision=6, linewidth=120, sci_mode=False)

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)

# 绘制ReLU函数图
# d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

# 绘制ReLU函数的导数
y.backward(torch.ones_like(x), retain_graph=True)
# d2l.plot(x.detach(), x.grad, 'x', 'grad(x)', figsize=(5, 2.5))

# 绘制sigmoid函数
y = torch.sigmoid(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

# 绘制sigmoid函数的导数
x.grad.data.zero_()  # 清除以前的梯度
y.backward(torch.ones_like(x), retain_graph=True)
# d2l.plot(x.detach(), x.grad, 'x', 'grad(x)', figsize=(5, 2.5))

# 绘制tanh函数
y = torch.tanh(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

# 绘制tanh函数的导数
x.grad.data.zero_() # 清除以前的梯度
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad(x)', figsize=(5, 2.5))

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
torch.set_printoptions(edgeitems=2, precision=6, linewidth=100, sci_mode=False)

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans,
                                                download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans,
                                               download=True)


# print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(images, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()
    return axes


# 测试显示几个样本
# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), num_rows=2, num_cols=9, titles=get_fashion_mnist_labels(y))

# 读取小批量数据
batch_size = 256


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

# 检查一下读取数据消耗的时间
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'read data using time {timer.stop()} sec')


def load_data_fashion_mnist(batch_size, resize=None):
    """用于获取和读取Fashion-MNIST数据集。 这个函数返回训练集和验证集的数据迭代器。

    Args:
        batch_size (int): 批量大小，每个小批量包含的样本数量。
        resize (int or tuple): 用来将图像大小调整为另一种形状。默认为None，表示不调整大小。
            如果是一个整数，图像将被调整为该整数表示的正方形大小。
            如果是一个元组，图像将被调整为元组表示的宽度和高度。

    Returns:
        tuple: 包含训练集和验证集的数据迭代器的元组。每个数据迭代器可以用于迭代获取小批量的数据。
    """
    # 将 transforms.ToTensor() 转换操作添加到变换列表中，以将图像数据转换为张量形式
    # 在 PyTorch 中，transforms.ToTensor() 是一个用于将 PIL 图像或 NumPy 数组转换为张量的变换操作。
    # 它将图像数据从原始的 PIL.Image 格式或 NumPy 数组格式转换为 PyTorch 张量格式。转换后的张量将具有与原始图像相同
    # 的形状，并且像素值将被标准化到 [0, 1] 范围内。
    #
    # transforms.ToTensor() 操作的输出是一个三维张量，如果原始图像是灰度图像，则形状为 (C, H, W)，
    # 其中 C 是通道数，H 和 W 分别是高度和宽度。如果原始图像是彩色图像，则形状为 (3, H, W)，
    # 其中 3 表示红、绿、蓝三个通道。
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    # 将变换操作列表 trans 组合成一个可用的转换操作，以便在数据加载过程中一次性应用于数据
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers())
            )


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

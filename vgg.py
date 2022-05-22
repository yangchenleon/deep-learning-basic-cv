import torch
import torchvision
from torchvision.datasets import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn, optim
from utils import *


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 每运行一次面积缩小一半
    return nn.Sequential(*layers)


if __name__ == '__main__':
    # load data
    data_path = "./data"
    batch_size = 128
    num_workers = 4
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(224)])
    mnist_train = FashionMNIST(root=data_path, train=True, download=True, transform=trans)
    mnist_test = FashionMNIST(root=data_path, train=False, download=True, transform=trans)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # models define
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    ratio = 4
    # 减小各VGG块输出通道数，缩小模型复杂度
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    # 加入全连接层部分
    net = nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 25088 = out_channels * 7 * 7
        nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

    # parameter, device, optimizer, loss
    num_epochs, lr = 10, 0.05
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    # training and evaluation
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for X, y in tqdm(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            ls = loss(net(X), y)
            ls.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(ls * X.shape[0], accuracy(net(X), y), X.shape[0])
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    torch.save(net.state_dict(), './models/vgg.pth')

    # predict
    model = net
    model.load_state_dict(torch.load('./models/vgg.pth', map_location=device))
    X, y = next(iter(DataLoader(mnist_test, batch_size=8, shuffle=True)))
    trues = get_fashion_mnist_labels(y)
    with torch.no_grad():
        preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    X = torchvision.transforms.Resize([28, 28])(X)
    n = 8
    show_images(X[0:n].reshape((n, 28, 28)), 2, n // 2, titles=titles[0:n], scale=2)

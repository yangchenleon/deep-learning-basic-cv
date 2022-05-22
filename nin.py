import torchvision
from torchvision.datasets import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn, optim
from utils import *


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)), nn.ReLU()
    )


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
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0), nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2), nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1), nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        # 标签类别数是10
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
        nn.Flatten()
    )

    # parameter, device, optimizer, loss
    num_epochs, lr = 10, 0.1
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
    torch.save(net.state_dict(), './models/nin.pth')

    # predcit
    model = net
    model.load_state_dict(torch.load('./models/nin.pth', map_location=device))
    X, y = next(iter(DataLoader(mnist_test, batch_size=18, shuffle=True)))
    trues = get_fashion_mnist_labels(y)
    with torch.no_grad():
        preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    X = torchvision.transforms.Resize([28, 28])(X)
    n = 8
    show_images(X[0:n].reshape((n, 28, 28)), 2, n // 2, titles=titles[0:n], scale=2)

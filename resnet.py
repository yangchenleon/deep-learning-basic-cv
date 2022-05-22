from torch import *
import torchvision
from torchvision.datasets import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn, optim
from utils import *


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=(3, 3), padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=(1, 1),
                               stride=strides) if not use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


if __name__ == '__main__':
    # load data
    data_path = "./data"
    batch_size = 256
    num_workers = 4
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(224)])
    mnist_train = FashionMNIST(root=data_path, train=True, download=True, transform=trans)
    mnist_test = FashionMNIST(root=data_path, train=False, download=True, transform=trans)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # models define
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))

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
    torch.save(net.state_dict(), './models/sresnet.pth')

    # predict
    model = net
    model.load_state_dict(torch.load('./models/resnet.pth', map_location=device))
    X, y = next(iter(DataLoader(mnist_test, batch_size=18, shuffle=True)))
    trues = get_fashion_mnist_labels(y)
    with torch.no_grad():
        preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    X = torchvision.transforms.Resize([28, 28])(X)
    n = 8
    show_images(X[0:n].reshape((n, 28, 28)), 2, n // 2, titles=titles[0:n], scale=2)

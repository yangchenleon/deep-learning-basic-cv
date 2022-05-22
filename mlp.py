import torch
import torchvision
from torchvision.datasets import *
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from utils import *


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':
    # load data
    data_path = "./data"
    batch_size = 256
    num_workers = 4
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])
    mnist_train = FashionMNIST(root=data_path, train=True, download=True, transform=trans)
    mnist_test = FashionMNIST(root=data_path, train=False, download=True, transform=trans)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # define Multilayer Perceptron models
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    # parameter, device, optimizer, loss
    lr, num_epochs = 0.1, 10
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    # train the models
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
    torch.save(net.state_dict(), './models/mlp.pth')

    # predict
    model = net
    model.load_state_dict(torch.load('./models/mlp.pth', map_location=device))
    X, y = next(iter(DataLoader(mnist_test, batch_size=18, shuffle=True)))
    trues = get_fashion_mnist_labels(y)
    with torch.no_grad():
        preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    n = 8
    show_images(X[0:n].reshape((n, 28, 28)), 2, n // 2, titles=titles[0:n], scale=2)

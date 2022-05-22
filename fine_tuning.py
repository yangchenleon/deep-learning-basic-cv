import os
import torch
import torchvision
from torchvision.datasets import *
from torch.utils.data import DataLoader
from torch import nn, optim
from utils import *
from tqdm import tqdm

if __name__ == '__main__':
    # data preprocess
    data_path = './data/hotdog'
    batch_size = 128
    model_dir = 'E:/cache/torch'
    normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize])

    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize])

    train_imgs = ImageFolder(os.path.join(data_path, 'train'), transform=train_augs)
    test_imgs = ImageFolder(os.path.join(data_path, 'test'), transform=test_augs)

    train_iter = DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_imgs, batch_size=batch_size)

    # model load
    os.environ['TORCH_HOME'] = model_dir  # setting the environment variable
    net = torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 2)

    # parameter, device, optimizer, loss
    num_epochs, lr = 5, 1e-2
    param_group = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    nn.init.xavier_uniform_(net.fc.weight)
    print('training on', device)
    net.to(device)
    loss = nn.CrossEntropyLoss()
    if param_group:
        params_1x = [param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
        optimizer = optim.SGD([{'params': params_1x}, {'params': net.fc.parameters(), 'lr': lr * 10}],
                              lr=lr, weight_decay=0.001)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.001)

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
    torch.save(net.state_dict(), './models/fine_tuning_on_resnet18.pth')

    # save models
    model = net
    model.load_state_dict(torch.load('./models/fine_tuning_on_resnet18.pth', map_location=device))
    idx = get_rand_index(len(test_imgs), 8)
    imgs = [ImageFolder(os.path.join(data_path, 'test'))[i][0] for i in idx]
    X = torch.stack([test_imgs[i][0] for i in idx], dim=0)
    y = [test_imgs[i][1] for i in idx]
    trues = get_hotdog_labels(y)
    with torch.no_grad():
        preds = get_hotdog_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    n = 8
    # 对于tensor的图形，3*224*224需要变成224*224*3, PIL不需要
    # img = X.swapaxes(1,2)
    # img = img.swapaxes(2,3)
    show_images(imgs[0:n], 2, n // 2, titles=titles[0:n], scale=2)

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch import optim


def cifar10loaders(train_batch_size=128, test_batch_size=10):

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def adjust_learning_rate(optimizer, epoch, epoch_thresholds=[60, 120, 160]):

    if epoch == epoch_thresholds[0]:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / 5
    elif epoch == epoch_thresholds[1]:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / 5
    elif epoch == epoch_thresholds[2]:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / 5

    return optimizer

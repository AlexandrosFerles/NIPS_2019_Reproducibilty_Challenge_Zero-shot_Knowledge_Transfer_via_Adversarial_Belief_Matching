import collections
import json
import torch
import torchvision
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler


# Borrowed from https://github.com/ozan-oktay/Attention-Gated-Networks


def json_file_to_pyobj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json2obj(open(filename).read())


def dataset_transforms(dataset_name):

    if dataset_name == 'cifar10':
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

        return transform_train, transform_test

    elif dataset_name == 'svhn':

        normalize = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))

        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        return transform


def cifar10loaders(train_batch_size=128, test_batch_size=10):

    transform_train, transform_test = dataset_transforms('cifar10')

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader


def svhnLoaders(train_batch_size=128, test_batch_size=10):

    transform = dataset_transforms('svhn')

    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=0)

    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader


def cifar10loadersM(M, train_batch_size=128, test_batch_size=10):

    transform_train, transform_test = dataset_transforms('cifar10')

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    temp_trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

    # sample M data per_class
    data_collected = [0]*10
    total_collected = 0
    success = 10*M
    indices = []
    for index, (_, label) in temp_trainloader:
        if data_collected[label] < M:
            data_collected[label] += 1
            indices.append(index)
            total_collected += 1
        if total_collected == success:
            break

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, sampler=SubsetRandomSampler(indices))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader


def svhnloadersM(M, train_batch_size=128, test_batch_size=10):

    transform = dataset_transforms('svhn')

    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    temp_trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)

    # sample M data per_class
    data_collected = [0] * 10
    total_collected = 0
    success = 10 * M
    indices = []
    for index, (_, label) in temp_trainloader:
        if data_collected[label] < M:
            data_collected[label] += 1
            indices.append(index)
            total_collected += 1
        if total_collected == success:
            break

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, sampler=SubsetRandomSampler(indices))

    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=0)

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


def attention_loss(att1, att2):
    # derive l2 norm of each attention map
    att1_norm = F.normalize(att1.pow(2).mean(1).view(x.size(0), -1))
    att2_norm = F.normalize(att2.pow(2).mean(1).view(x.size(0), -1))

    # Loss now is just the p2-norm of the normalized attention maps!
    loss = (att1_norm - att2_norm).pow(2).mean()
    return loss


def kd_att_loss(student_outputs, teacher_outputs, labels, T=4, a=0.9, b=1000, criterion1=nn.CrossEntropyLoss(), criterion2=nn.KLDivLoss()):

    student_out, student_activations = student_outputs[0], student_outputs[1:]
    teacher_out, teacher_activations = teacher_outputs[0], teacher_outputs[1:]

    activation_pairs = zip(student_activations, teacher_activations)

    loss_term1 = (1-a) * criterion1(student_out, labels)
    # changed to log softmax for student_out and 2a for loss_term2 after inspection of the official code
    loss_term2 = criterion2(F.log_softmax(student_out/T), F.softmax(teacher_out/T))
    loss_term2 *= (T**2)*2*a
    attention_losses = [attention_loss(att1, att2) for (att1, att2) in activation_pairs]
    loss_term3 = b * sum(attention_losses)

    return loss_term1 + loss_term2 + loss_term3


def generator_loss(student_outputs, teacher_outputs, criterion=nn.KLDivLoss(), T=1):

    gen_loss = -criterion(F.log_softmax(student_outputs / T), F.softmax(teacher_outputs / T))
    return gen_loss


def student_loss_zero_shot(student_outputs, teacher_outputs, b):

    student_out, student_activations = student_outputs[0], student_outputs[1:]
    teacher_out, teacher_activations = teacher_outputs[0], teacher_outputs[1:]

    activation_pairs = zip(student_activations, teacher_activations)

    attention_losses = [attention_loss(att1, att2, T=1) for (att1, att2) in activation_pairs]
    loss_term1 = b * sum(attention_losses)
    loss = loss_term1 - generator_loss(student_out, teacher_out)

    return loss

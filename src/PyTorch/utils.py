import collections
import json
import torch
import torchvision
from torch import nn as nn
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader


def svhnLoaders(train_batch_size=128, test_batch_size=10):
    transform = dataset_transforms('svhn')

    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=4)

    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader


def cifar10loadersM(M, train_batch_size=128, test_batch_size=10, apply_test=False):
    transform_train, transform_test = dataset_transforms('cifar10')

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    temp_trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

    # sample M data per_class
    data_collected = [0] * 10
    total_collected = 0
    success = 10 * M
    indices = []
    for index, (_, label) in enumerate(temp_trainloader):
        if data_collected[label] < M:
            data_collected[label] += 1
            indices.append(index)
            total_collected += 1
        if total_collected == success:
            break

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              sampler=SubsetRandomSampler(indices))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    if apply_test:

        data_collected = [0] * 10
        total_collected = 0
        indices = []
        for index, (_, label) in enumerate(testloader):
            if data_collected[label] < M:
                data_collected[label] += 1
                indices.append(index)
                total_collected += 1
            if total_collected == success:
                break

        new_testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                     sampler=SubsetRandomSampler(indices))
        return trainloader, new_testloader

    else:
        return trainloader, testloader


def svhnloadersM(M, train_batch_size=128, test_batch_size=10, apply_test=False):
    transform = dataset_transforms('svhn')

    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    temp_trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    # sample M data per_class
    data_collected = [0] * 10
    total_collected = 0
    success = 10 * M
    indices = []
    for index, (_, label) in enumerate(temp_trainloader):
        if data_collected[label] < M:
            data_collected[label] += 1
            indices.append(index)
            total_collected += 1
        if total_collected == success:
            break

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              sampler=SubsetRandomSampler(indices))

    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    if apply_test:

        data_collected = [0] * 10
        total_collected = 0
        indices = []
        for index, (_, label) in enumerate(testloader):
            if data_collected[label] < M:
                data_collected[label] += 1
                indices.append(index)
                total_collected += 1
            if total_collected == success:
                break

        new_testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                     sampler=SubsetRandomSampler(indices))
        return trainloader, new_testloader

    else:
        return trainloader, testloader
    
    
def get_matching_indices(dataset, teacher, student, device, n=1000):
    
    if dataset.lower() == 'cifar10':
        _, transform_test = dataset_transforms('cifar10')
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset.lower() == 'svhn':
        _, transform_test = dataset_transforms('svhn')
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)

    total_collected = 0
    indices = []

    teacher.eval()
    student.eval()
    for index, (images, label) in enumerate(testloader):

        images = images.to(device)
        labels = labels.to(device)

        student_outputs = student(images)[0]
        _, student_predicted = torch.max(student_outputs.data, 1)

        teacher_outputs = teacher(images)[0]
        _, teacher_predicted = torch.max(teacher_outputs.data, 1)

        if student_predicted == teacher_predicted:
            indices.append(index)
            total_collected += 1

        if total_collected == n:
            break

    new_testloader = torch.utils.data.DataLoader(testset, batch_size=1, sampler=SubsetRandomSampler(indices))

    return new_testloader


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
    att1_norm = F.normalize(att1.pow(2).mean(1).view(att1.size(0), -1))
    att2_norm = F.normalize(att2.pow(2).mean(1).view(att2.size(0), -1))

    # Loss now is just the p2-norm of the normalized attention maps!
    loss = (att1_norm - att2_norm).pow(2).mean()
    return loss


def kd_att_loss(student_outputs, teacher_outputs, labels, T=4, a=0.9, b=1000, criterion1=nn.CrossEntropyLoss(),
                criterion2=nn.KLDivLoss()):
    student_out, student_activations = student_outputs[0], student_outputs[1:]
    teacher_out, teacher_activations = teacher_outputs[0], teacher_outputs[1:]

    activation_pairs = zip(student_activations, teacher_activations)

    loss_term1 = (1 - a) * criterion1(student_out, labels)
    # changed to log softmax for student_out and 2a for loss_term2 after inspection of the official code
    loss_term2 = criterion2(F.log_softmax(student_out / T, dim=1), F.softmax(teacher_out / T, dim=1))
    loss_term2 *= (T ** 2) * 2 * a
    attention_losses = [attention_loss(att1, att2) for (att1, att2) in activation_pairs]
    loss_term3 = b * sum(attention_losses)

    return loss_term1 + loss_term2 + loss_term3


def generator_loss(student_outputs, teacher_outputs, criterion=nn.KLDivLoss(), T=1):
    gen_loss = -criterion(F.log_softmax(student_outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1))

    return gen_loss


def student_loss_zero_shot(student_outputs, teacher_outputs, b=250):
    student_out, student_activations = student_outputs[0], student_outputs[1:]
    teacher_out, teacher_activations = teacher_outputs[0], teacher_outputs[1:]

    activation_pairs = zip(student_activations, teacher_activations)

    attention_losses = [attention_loss(att1, att2) for (att1, att2) in activation_pairs]
    loss_term1 = b * sum(attention_losses)
    loss = loss_term1 - generator_loss(student_out, teacher_out)

    return loss


def plot_samples_from_generator():
    from Generator import Generator
    import numpy as np
    import torchvision.utils as utils

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    generator_net = Generator()
    generator_net = generator_net.to(device)

    checkpoints = [
        'checkpoints/CIFAR10/Zero-Shot/ours_zero_shot_teacher_wrn-40-2_student_wrn-16-1-M-0-seed-0-CIFAR10-generator-dict.pth',
        'checkpoints/CIFAR10/Zero-Shot/ours_zero_shot_teacher_wrn-40-2_student_wrn-40-1-M-0-seed-1-CIFAR10-generator-dict.pth',
        'checkpoints/CIFAR10/Zero-Shot/reproducibility_zero_shot_teacher_wrn-40-1_student_wrn-16-2-M-0-seed-2-CIFAR10-generator-dict.pth',
        'checkpoints/CIFAR10/Zero-Shot/reproducibility_zero_shot_teacher_wrn-16-2_student_wrn-16-1-M-0-seed-0-CIFAR10-generator-dict.pth']

    images = []
    for checkpoint in checkpoints:
        torch_checkpoint = torch.load(checkpoint, map_location=device)
        generator_net.load_state_dict(torch_checkpoint)
        print('passed checkpoint')
        for i in np.arange(0, 10):
            z = torch.randn((128, 100)).to(device)
            sample = generator_net(z)

            # convert image to [0, 1]
            # image = np.transpose(sample[0].detach().numpy(), [1,2,0])
            image = sample[0].detach().numpy()
            image = (image - image.min()) / (image.max() - image.min())

            images.append(image)

    utils.save_image(torch.Tensor(np.array(images)), 'generator_samples.png', 10)


def plot_performance_for_models(no_teacher, kd_att, kd_att_full, zero_shot, vid):
    import matplotlib.pyplot as plt
    import numpy as np
    x = [0, 10, 25, 50, 75, 100]

    fig = plt.figure()
    ax = fig.gca()

    ax.scatter(0, zero_shot[0], marker='*', color='b', linestyle='--', s=150)
    plt.plot(x, no_teacher, color='g', marker='o', markersize=3)
    plt.plot(x, kd_att, marker='o', color='r', markersize=3)
    plt.plot(x, kd_att_full, color='pink')
    plt.plot(0, zero_shot[0], color='b', linestyle='--', marker='o', markersize=3)
    # plt.plot(x, zero_shot, color='b', linestyle='--', marker='o', markersize=3)
    plt.plot(100, vid, marker='*', color='gold')

    plt.yticks(np.arange(0, 101, step=10))
    plt.xlabel('M')
    plt.ylabel('test accuracy(%)')
    plt.legend(['No Teacher', 'KD+AT', 'KD+AT full data', 'Zero-Shot', 'vid'], loc='lower right')

    plt.show()


def plot_cifar():
    # kd att is missing and zero shot with M
    no_teacher = [10, 23.7, 34.4, 41.69, 54.45, 57.02]
    kd_att = [10, 36.96, 60.05, 0, 73.84, 76.67]
    kd_att_full = [92.15, 92.15, 92.15, 92.15, 92.15, 92.15]
    zero_shot = [84.02, 84.02, 84.02, 84.02, 84.02, 84.02]
    vid = [81.59]

    plot_performance_for_models(no_teacher, kd_att, kd_att_full, zero_shot, vid)


def plot_svhn():
    # only zero shot with M

    no_teacher = [10, 11.97, 31.83, 44.08, 50.07, 56.71]
    kd_att = [10, 37.35, 48.71, 68.84, 78.51, 81.18]
    kd_att_full = [95.19, 95.19, 95.19, 95.19, 95.19, 95.19]
    zero_shot = [94.21, 94.21, 94.21, 94.21, 94.21, 94.21]

    plot_performance_for_models(no_teacher, kd_att, kd_att_full, zero_shot)

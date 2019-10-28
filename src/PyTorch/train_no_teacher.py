import torch
from torch import nn as nn
from torch import optim
import numpy as np
from utils import json_file_to_pyobj
from WideResNet import WideResNet
from utils import adjust_learning_rate, kd_att_loss
from train_scratches import set_seed
import os


def _test_set_eval(net, device, test_loader):

    with torch.no_grad():

        correct, total = 0, 0
        net.eval()

        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        accuracy = round(100 * accuracy, 2)

        return accuracy


def _train_seed_no_teacher(net, M, loaders, device, log=False, checkpoint=False, logfile='', checkpointFile=''):

    train_loader, test_loader = loaders
    # or 50000 / (10*M) since M is sample per each one of 10 classes
    epochs = int(200 * (5000 / M))
    epoch_thresholds = [int(x) for x in [0.3*epochs, 0.6*epochs, 0.8*epochs]]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    best_test_set_accuracy = 0

    for epoch in range(epochs):

        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            student_outputs = net(inputs)

            loss = criterion(student_outputs, labels)
            loss.backward()
            optimizer.step()

        optimizer = adjust_learning_rate(optimizer, epoch + 1, epoch_thresholds=epoch_thresholds)
        epoch_accuracy = _test_set_eval(net, device, test_loader)

        if log:
            with open(os.path.join('./', logfile), "a") as temp:
                temp.write('Accuracy at epoch {} is {}%\n'.format(epoch + 1, epoch_accuracy))

        if epoch_accuracy > best_test_set_accuracy:
            best_test_set_accuracy = epoch_accuracy
            if checkpoint:
                torch.save(net.state_dict(), checkpointFile)

    return best_test_set_accuracy


def train(args):

    json_options = json_file_to_pyobj(args.config)
    kd_att_configurations = json_options.training

    wrn_depth = kd_att_configurations.wrn_depth
    wrn_width = kd_att_configurations.wrn_width

    M = kd_att_configurations.M

    dataset = kd_att_configurations.dataset
    seeds = [int(seed) for seed in kd_att_configurations.seeds]
    log = bool(kd_att_configurations.checkpoint)

    if log:
        net_str = "WideResNet-{}-{}".format(wrn_depth, wrn_width)
        logfile = "No_Teacher-{}-{}-M-{}.txt".format(net_str, kd_att_configurations.dataset, M)
        with open(os.path.join('./', logfile), "w") as temp:
            temp.write('No teacher {} in {} with M={}\n'.format(net_str, kd_att_configurations.dataset, M))
    else:
        logfile = ''

    checkpoint = bool(kd_att_configurations.checkpoint)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    test_set_accuracies = []

    for seed in seeds:

        set_seed(seed)

        if dataset.lower() == 'cifar10':

            # Full data
            if M == 5000:
                from utils import cifar10loaders
                loaders = cifar10loaders()
            # No data
            elif M == 0:
                from utils import cifar10loaders
                _, test_loader = cifar10loaders()
            else:
                from utils import cifar10loadersM
                loaders = cifar10loadersM(M)

        elif dataset.lower() == 'svhn':

            # Full data
            if M == 5000:
                from utils import svhnLoaders
                loaders = svhnLoaders()
            # No data
            elif M == 0:
                from utils import svhnLoaders
                _, test_loader = svhnLoaders()
            else:
                from utils import svhnloadersM
                loaders = svhnloadersM(M)

        else:
            raise ValueError('Datasets to choose from: CIFAR10 and SVHN')

        if log:
            with open(os.path.join('./', logfile), "a") as temp:
                temp.write('------------------- SEED {} -------------------\n'.format(seed))

        strides = [1, 1, 2, 2]

        net = WideResNet(d=wrn_depth, k=wrn_width, n_classes=10, input_features=3, output_features=16, strides=strides)
        net = net.to(device)

        checkpointFile = 'No_teacher_wrn-{}-{}-M-{}-seed-{}-{}-dict.pth'.format(wrn_depth, wrn_width, M, seed, dataset) if checkpoint else ''

        best_test_set_accuracy = _train_seed_no_teacher(net, M, loaders, device, log, checkpoint, logfile, checkpointFile)

        if log:
            with open(os.path.join('./', logfile), "a") as temp:
                temp.write('Best test set accuracy of seed {} is {}\n'.format(seed, best_test_set_accuracy))

        test_set_accuracies.append(best_test_set_accuracy)

        if log:
            with open(os.path.join('./', logfile), "a") as temp:
                temp.write('Best test set accuracy of seed {} is {}\n'.format(seed, best_test_set_accuracy))

    mean_test_set_accuracy, std_test_set_accuracy = np.mean(test_set_accuracies), np.std(test_set_accuracies)

    if log:
        with open(os.path.join('./', logfile), "a") as temp:
            temp.write('Mean test set accuracy is {} with standard deviation equal to {}\n'.format(mean_test_set_accuracy, std_test_set_accuracy))

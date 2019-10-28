import os
import torch
from torch import optim
import torch.nn as nn
import numpy as np
import random
from utils import json_file_to_pyobj
from WideResNet import WideResNet
from utils import adjust_learning_rate
from tqdm import tqdm


def set_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _train_seed(net, loaders, device, log=False, checkpoint=False, logfile='', checkpointFile=''):
    train_loader, test_loader = loaders
    epochs = 200

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)

    best_test_set_accuracy = 0

    for epoch in tqdm(range(epochs)):

        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            wrn_outputs = net(inputs)
            outputs = wrn_outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        optimizer = adjust_learning_rate(optimizer, epoch + 1)

        if epoch >= 160:
            with torch.no_grad():

                correct = 0
                total = 0

                net.eval()
                for data in test_loader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)

                    wrn_outputs = net(images)
                    outputs = wrn_outputs[0]
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                epoch_accuracy = correct / total
                epoch_accuracy = round(100 * epoch_accuracy, 2)

                if log:
                    with open(logfile, 'a') as temp:
                        temp.write('Accuracy at epoch {} is {}%\n'.format(epoch + 1, epoch_accuracy))

                if epoch_accuracy > best_test_set_accuracy:
                    best_test_set_accuracy = epoch_accuracy
                    if checkpoint:
                        torch.save(net.state_dict(), checkpointFile)

    return best_test_set_accuracy


def train(args):
    json_options = json_file_to_pyobj(args.config)
    training_configurations = json_options.training

    wrn_depth = training_configurations.wrn_depth
    wrn_width = training_configurations.wrn_width
    dataset = training_configurations.dataset.lower()
    seeds = [int(seed) for seed in training_configurations.seeds]
    log = bool(training_configurations.checkpoint)

    if log:
        logfile = training_configurations.logfile
        with open(logfile, 'w') as temp:
            temp.write('WideResNet-{}-{} scratch in {}\n'.format(wrn_depth, wrn_width, training_configurations.dataset))
    else:
        logfile = ''

    checkpoint = bool(training_configurations.checkpoint)

    if dataset == 'cifar10':

        from utils import cifar10loaders
        loaders = cifar10loaders()

    elif dataset == 'svhn':

        from utils import svhnLoaders
        loaders = svhnLoaders()
    else:
        ValueError('Datasets to choose from: CIFAR10 and SVHN')

    if torch.cuda.is_available():
        device = torch.device('cuda:2')
    else:
        device = torch.device('cpu')

    test_set_accuracies = []

    for seed in seeds:
        set_seed(seed)

        if log:
            with open(logfile, 'a') as temp:
                temp.write('------------------- SEED {} -------------------\n'.format(seed))

        strides = [1, 1, 2, 2]
        net = WideResNet(d=wrn_depth, k=wrn_width, n_classes=10, input_features=3, output_features=16, strides=strides)
        net = net.to(device)

        checkpointFile = '_wrn-{}-{}-seed-{}-{}-dict.pth'.format(wrn_depth, wrn_width, dataset, seed) if checkpoint else ''
        best_test_set_accuracy = _train_seed(net, loaders, device, log, checkpoint, logfile, checkpointFile)

        if log:
            with open(logfile, 'a') as temp:
                temp.write('Best test set accuracy of seed {} is {}\n'.format(seed, best_test_set_accuracy))

        test_set_accuracies.append(best_test_set_accuracy)

        if log:
            with open(logfile, 'a') as temp:
                temp.write('Best test set accuracy of seed {} is {}\n'.format(seed, best_test_set_accuracy))

    mean_test_set_accuracy, std_test_set_accuracy = np.mean(test_set_accuracies), np.std(test_set_accuracies)

    if log:
        with open(logfile, 'a') as temp:
            temp.write('Mean test set accuracy is {} with standard deviation equal to {}\n'.format(mean_test_set_accuracy, std_test_set_accuracy))


if __name__ == '__main__':
    import argparse

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    parser = argparse.ArgumentParser(description='WideResNet Scratches')

    parser.add_argument('-config', '--config', help='Training Configurations', required=True)

    args = parser.parse_args()

    train(args)
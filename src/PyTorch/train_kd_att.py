import os
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from utils import json_file_to_pyobj
from WideResNet import WideResNet
from utils import adjust_learning_rate_scratch

from train_scratches import set_seed


def attention_loss(att1, att2) :
    # TODO: Implement the attention loss between the pairs
    loss = 0
    return loss


def kd_att_loss(student_outputs, teacher_outputs, labels, T=4, a=0.9, b=1000, criterion1=nn.CrossEntropyLoss(), criterion2=nn.KLDivLoss()):

    student_out, student_activations = student_outputs[0], student_outputs[1:]
    teacher_out, teacher_activations = teacher_outputs[0], teacher_outputs[1:]

    activation_pairs = zip(student_activations, teacher_activations)

    loss_term1 = (1-a) * criterion1(student_out, labels)
    loss_term2 = criterion2(F.softmax(student_out/T), F.softmax(teacher_out/T))
    attention_losses = [attention_loss(att1, att2) for (att1, att2) in activation_pairs]
    loss_term3 = (b/2) * sum(attention_losses)

    return loss_term1 + loss_term2 + loss_term3


def _train_seed_kd_att(teacher_net, student_net, M, loaders, device, log=False, checkpoint=False, logfile='', checkpointFile=''):

    train_loader, test_loader = loaders
    epochs = 200 * (50000 / M)
    epoch_thresholds = [int(x) for x in [0.3*epochs, 0.6*epochs, 0.8*epochs]]

    optimizer = optim.SGD(student_net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)

    best_test_set_accuracy = 0
    teacher_net.eval()

    for epoch in range(epochs):

        student_net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            student_outputs = student_net(inputs)
            teacher_outputs = teacher_net(inputs)

            loss = kd_att_loss(student_outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()

        optimizer = adjust_learning_rate_scratch(optimizer, epoch + 1, epoch_thresholds=epoch_thresholds)

        with torch.no_grad():

            correct = 0
            total = 0

            student_net.eval()
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                student_net(images)
                outputs = student_net(images)[0]
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
                    torch.save(student_net.state_dict(), checkpointFile)

    return best_test_set_accuracy


def train(args):

    json_options = json_file_to_pyobj(args.config)
    kd_att_configurations = json_options.kd_att

    wrn_depth_teacher = kd_att_configurations.wrn_depth_teacher
    wrn_width_teacher = kd_att_configurations.wrn_width_teacher
    wrn_depth_student = kd_att_configurations.wrn_depth_student
    wrn_width_student = kd_att_configurations.wrn_width_student

    M = kd_att_configurations.M

    dataset = kd_att_configurations.dataset.lower()
    seeds = [int(seed) for seed in kd_att_configurations.seeds]
    log = bool(kd_att_configurations.checkpoint)

    if log:
        logfile = kd_att_configurations.logfile
        with open(logfile, 'w') as temp:
            teacher_str = 'WideResNet-{}-{}'.format(wrn_depth_teacher, wrn_width_teacher)
            student_str = 'WideResNet-{}-{}'.format(wrn_depth_student, wrn_width_student)
            temp.write('KD_ATT with teacher {} and student {} in {}\n'.format(teacher_str, student_str, kd_att_configurations.dataset))
    else:
        logfile = ''

    checkpoint = bool(kd_att_configurations.checkpoint)

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

        teacher_net = WideResNet(d=wrn_depth_teacher, k=wrn_width_teacher, n_classes=10, input_features=3, output_features=16, strides=strides)
        teacher_net = teacher_net.to(device)
        temp_dataset_name = 'CIFAR10' if dataset =='cifar10' else 'SVHN'
        # TODO: Needs fix for SVHN!
        torch_checkpoint = torch.load('./PreTrainedModels/PreTrainedScratches/{}/wrn-{}-{}-seed-{}-dict.pth'.format(temp_dataset_name, wrn_depth_teacher, wrn_width_teacher, seed))
        teacher_net.load_state_dict(torch_checkpoint)
        
        student_net = WideResNet(d=wrn_depth_student, k=wrn_width_student, n_classes=10, input_features=3, output_features=16, strides=strides)
        student_net = student_net.to(device)

        checkpointFile = 'kd_att_teacher_wrn-{}-{}_student_wrn-{}-{}-seed-{}-{}-dict.pth'.format(wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, dataset, seed) if checkpoint else ''
        best_test_set_accuracy = _train_seed_kd_att(teacher_net, student_net, M, loaders, device, log, checkpoint, logfile, checkpointFile)

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
            temp.write(
                'Mean test set accuracy is {} with standard deviation equal to {}\n'.format(mean_test_set_accuracy,
                                                                                            std_test_set_accuracy))

import os
import torch
from torch import optim
import numpy as np
from utils import json_file_to_pyobj
from WideResNet import WideResNet
from utils import kd_att_loss
from copy import deepcopy
from train_scratches import set_seed
from tqdm import tqdm


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


def _loss_with_weight_regularization(student_outputs, teacher_outputs, labels, student, clone, lamda=1):

    kd_at_loss = kd_att_loss(student_outputs, teacher_outputs, labels, b=0)

    acc = 0
    for clone_param, student_param in zip(clone.parameters(), student.parameters()):
        acc += torch.mean(torch.abs(clone_param - student_param))

    return kd_at_loss + lamda*acc


def _train_extra_M(epochs, teacher_net, student_net, M, loaders, device, log=False, checkpoint=False, logfile='', checkpointFile='', finalCheckpointFile='', genCheckpointFile=''):

    train_loader, test_loader = loaders
    best_test_set_accuracy = _test_set_eval(student_net, device, test_loader)
    if log:
        with open(logfile, 'a') as temp:
            temp.write('Initial test accuracy is {}\n'.format(best_test_set_accuracy))

    student_optimizer = optim.Adam(student_net.parameters(), lr=5e-4)
    cosine_annealing_student = optim.lr_scheduler.CosineAnnealingLR(student_optimizer, epochs)

    clone = deepcopy(student_net)

    teacher_net.eval()
    student_net.eval()

    for epoch in tqdm(range(epochs)):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            student_optimizer.zero_grad()

            student_outputs = student_net(inputs)
            teacher_outputs = teacher_net(inputs)
            loss = _loss_with_weight_regularization(student_outputs, teacher_outputs, labels, student_net, clone, lamda=1)
            loss.backward()
            student_optimizer.step()

        accuracy = _test_set_eval(student_net, device, test_loader)

        if accuracy > best_test_set_accuracy:
            best_test_set_accuracy = accuracy
            if log:
                with open(logfile, 'a') as temp:
                    temp.write('New best test accuracy is {}\n'.format(best_test_set_accuracy))

            if checkpoint:
                torch.save(student_net.state_dict(), checkpointFile)

        cosine_annealing_student.step()

    return best_test_set_accuracy


def train(args):

    json_options = json_file_to_pyobj(args.config)
    extra_M_configuration = json_options.training

    wrn_depth_teacher = extra_M_configuration.wrn_depth_teacher
    wrn_width_teacher = extra_M_configuration.wrn_width_teacher
    wrn_depth_student = extra_M_configuration.wrn_depth_student
    wrn_width_student = extra_M_configuration.wrn_width_student

    M = extra_M_configuration.M

    dataset = extra_M_configuration.dataset
    seeds = [int(seed) for seed in extra_M_configuration.seeds]
    log = True if extra_M_configuration.log.lower() == 'True' else False

    if dataset.lower() == 'cifar10':
        epochs = 200
    elif dataset.lower() =='svhn':
        epochs=100
    else:
        raise ValueError('Unknown dataset')

    if log:
        teacher_str = 'WideResNet-{}-{}'.format(wrn_depth_teacher, wrn_width_teacher)
        student_str = 'WideResNet-{}-{}'.format(wrn_depth_student, wrn_width_student)
        logfile = 'Extra_M_samples_Reproducibility_Zero_Shot_Teacher-{}-Student-{}-{}-M-{}-Zero-Shot.txt'.format(teacher_str, student_str, extra_M_configuration.dataset, M)
        with open(logfile, 'w') as temp:
            temp.write('Zero-Shot with teacher {} and student {} in {} with M-{}\n'.format(teacher_str, student_str, extra_M_configuration.dataset, M))
    else:
        logfile = ''

    checkpoint = bool(extra_M_configuration.checkpoint)

    if torch.cuda.is_available():
        device = torch.device('cuda:2')
    else:
        device = torch.device('cpu')

    test_set_accuracies = []

    for seed in seeds:

        set_seed(seed)

        if dataset.lower() == 'cifar10':

             from utils import cifar10loadersM
             loaders = cifar10loadersM(M)

        elif dataset.lower() == 'svhn':

            from utils import svhnloadersM
            loaders = svhnloadersM(M)

        else:
            raise ValueError('Datasets to choose from: CIFAR10 and SVHN')

        if log:
            with open(logfile, 'a') as temp:
                temp.write('------------------- SEED {} -------------------\n'.format(seed))

        strides = [1, 1, 2, 2]

        teacher_net = WideResNet(d=wrn_depth_teacher, k=wrn_width_teacher, n_classes=10, input_features=3, output_features=16, strides=strides)
        teacher_net = teacher_net.to(device)

        if dataset.lower() == 'cifar10':
            torch_checkpoint = torch.load('./PreTrainedModels/PreTrainedScratches/CIFAR10/wrn-{}-{}-seed-{}-dict.pth'.format(wrn_depth_teacher, wrn_width_teacher, seed), map_location=device)
        elif dataset.lower() == 'svhn':
            torch_checkpoint = torch.load('./PreTrainedModels/PreTrainedScratches/SVHN/wrn-{}-{}-seed-svhn-{}-dict.pth'.format(wrn_depth_teacher, wrn_width_teacher, seed), map_location=device)
        else:
            raise ValueError('Dataset not found')

        teacher_net.load_state_dict(torch_checkpoint)

        student_net = WideResNet(d=wrn_depth_student, k=wrn_width_student, n_classes=10, input_features=3, output_features=16, strides=strides)
        student_net = student_net.to(device)

        if dataset.lower() == 'cifar10':
            torch_checkpoint = torch.load('./PreTrainedModels/Zero-Shot/CIFAR10/reproducibility_zero_shot_teacher_wrn-{}-{}_student_wrn-{}-{}-M-0-seed-{}-CIFAR10-dict.pth'.format(wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, seed), map_location=device)
        elif dataset.lower() == 'svhn':
            torch_checkpoint = torch.load('./PreTrainedModels/Zero-Shot/SVHN/reproducibility_zero_shot_teacher_wrn-{}-{}_student_wrn-{}-{}-M-0-seed-{}-SVHN-dict.pth'.format(wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, seed), map_location=device)
        else:
            raise ValueError('Dataset not found')

        student_net.load_state_dict(torch_checkpoint)

        if checkpoint:
            teacher_str = 'WideResNet-{}-{}'.format(wrn_depth_teacher, wrn_width_teacher)
            student_str = 'WideResNet-{}-{}'.format(wrn_depth_student, wrn_width_student)
            checkpointFile = 'Checkpoint_Extra_M_samples_Reproducibility_Zero_Shot_Teacher-{}-Student-{}-{}-M-{}-Zero-Shot-seed-{}.pth'.format(teacher_str, student_str, extra_M_configuration.dataset, M, seed)
        else:
            checkpointFile = ''

        best_test_set_accuracy = _train_extra_M(epochs, teacher_net, student_net, M, loaders, device, log, checkpoint, logfile, checkpointFile)

        test_set_accuracies.append(best_test_set_accuracy)

        if log:
            with open(logfile, 'a') as temp:
                temp.write('Best test set accuracy of seed {} is {}\n'.format(seed, best_test_set_accuracy))

    mean_test_set_accuracy, std_test_set_accuracy = np.mean(test_set_accuracies), np.std(test_set_accuracies)

    if log:
        with open(logfile, 'a') as temp:
            temp.write(
                'Mean test set accuracy is {} with standard deviation equal to {}\n'.format(mean_test_set_accuracy, std_test_set_accuracy))


if __name__ == '__main__':

    import argparse

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    parser = argparse.ArgumentParser(description='WideResNet Scratches')

    parser.add_argument('-config', '--config', help='Training Configurations', required=True)

    args = parser.parse_args()

    train(args)

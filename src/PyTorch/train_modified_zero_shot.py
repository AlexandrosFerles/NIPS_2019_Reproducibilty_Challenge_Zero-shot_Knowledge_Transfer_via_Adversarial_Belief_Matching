import os
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from utils import json_file_to_pyobj
from WideResNet import WideResNet
from Generator import Generator
from utils import generator_loss, student_loss_zero_shot
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


def _train_seed_zero_shot(teacher_net, student_net, generator_net, loaders, device, log=False, checkpoint=False, logfile='', checkpointFile='', finalCheckpointFile='', genCheckpointFile=''):

    # Hardcoded values from paper and script training files of official GitHub repo!
    ng = 1
    ns = 10
    total_batches = 65000

    student_optimizer = optim.Adam(student_net.parameters(), lr=2e-3)
    cosine_annealing_student = optim.lr_scheduler.CosineAnnealingLR(student_optimizer, total_batches)
    generator_optimizer = optim.Adam(generator_net.parameters(), lr=1e-3)
    cosine_annealing_generator = optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, total_batches)

    best_test_set_accuracy = 0
    teacher_net.eval()

    for batch in tqdm(range(total_batches)):

        generator_net.train()
        # Hardcoded since batch size and noise dimension are constant

        for _ in range(ng):

            z = torch.randn((128, 100)).to(device)
            sample = generator_net(z)
            generator_optimizer.zero_grad()

            student_out = student_net(sample)[0]
            teacher_out = teacher_net(sample)[0]

            gen_loss = generator_loss(student_out, teacher_out)
            gen_loss.backward()
            # Added from official repo!
            torch.nn.utils.clip_grad_norm_(generator_net.parameters(), 5)
            generator_optimizer.step()

        student_net.train()
        for _ in range(ns):

            z = torch.randn((128, 100)).to(device)
            sample = generator_net(z)

            student_optimizer.zero_grad()

            student_outputs = student_net(sample)
            teacher_outputs = teacher_net(sample)

            student_loss = student_loss_zero_shot(student_outputs, teacher_outputs)
            student_loss.backward()
            # Likewise!
            torch.nn.utils.clip_grad_norm_(student_net.parameters(), 5)
            student_optimizer.step()

        test_loader = loaders

        if (batch+1) % 1000 == 0:
            batch_accuracy = _test_set_eval(student_net, device, test_loader)

            if log:
                with open(logfile, 'a') as temp:
                    temp.write('Accuracy at batch {} is {}%\n'.format(batch + 1, batch_accuracy))

            if batch_accuracy > best_test_set_accuracy:
                best_test_set_accuracy = batch_accuracy
                if checkpoint:
                    torch.save(student_net.state_dict(), checkpointFile)
                    torch.save(generator_net.state_dict(), genCheckpointFile)

        cosine_annealing_generator.step()
        cosine_annealing_student.step()

    torch.save(student_net.state_dict(), finalCheckpointFile)

    return best_test_set_accuracy


def train(args):

    json_options = json_file_to_pyobj(args.config)
    kd_att_configurations = json_options.training

    wrn_depth_teacher = kd_att_configurations.wrn_depth_teacher
    wrn_width_teacher = kd_att_configurations.wrn_width_teacher
    wrn_depth_student = kd_att_configurations.wrn_depth_student
    wrn_width_student = kd_att_configurations.wrn_width_student

    M = kd_att_configurations.M

    dataset = kd_att_configurations.dataset
    seeds = [int(seed) for seed in kd_att_configurations.seeds]
    log = bool(kd_att_configurations.checkpoint)

    if log:
        teacher_str = 'WideResNet-{}-{}'.format(wrn_depth_teacher, wrn_width_teacher)
        student_str = 'WideResNet-{}-{}'.format(wrn_depth_student, wrn_width_student)
        logfile = 'Teacher-{}-Student-{}-{}-M-{}-Zero-Shot.txt'.format(teacher_str, student_str, kd_att_configurations.dataset, M)
        with open(logfile, 'w') as temp:
            temp.write('Zero-Shot with teacher {} and student {} in {} with M-{}\n'.format(teacher_str, student_str, kd_att_configurations.dataset, M))
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

        generator_net = Generator()
        generator_net = generator_net.to(device)

        checkpointFile = 'zero_shot_teacher_wrn-{}-{}_student_wrn-{}-{}-M-{}-seed-{}-{}-dict.pth'.format(wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, M, seed, dataset) if checkpoint else ''
        finalCheckpointFile = 'zero_shot_teacher_wrn-{}-{}_student_wrn-{}-{}-M-{}-seed-{}-{}-final-dict.pth'.format(wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, M, seed, dataset) if checkpoint else ''
        genCheckpointFile = 'zero_shot_teacher_wrn-{}-{}_student_wrn-{}-{}-M-{}-seed-{}-{}-generator-dict.pth'.format(wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, M, seed, dataset) if checkpoint else ''

        best_test_set_accuracy = _train_seed_zero_shot(teacher_net, student_net, generator_net, test_loader, device, log, checkpoint, logfile, checkpointFile, finalCheckpointFile, genCheckpointFile)

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

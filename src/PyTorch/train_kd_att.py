import torch
from torch import optim
import numpy as np
from utils import json_file_to_pyobj
from WideResNet import WideResNet
from utils import adjust_learning_rate, kd_att_loss
from train_scratches import set_seed
import os
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


def _train_seed_kd_att(teacher_net, student_net, M, loaders, device, log=False, checkpoint=False, logfile='',
                       checkpointFile=''):
    train_loader, test_loader = loaders
    # or 50000 / (10*M) since M is sample per each one of 10 classes
    epochs = int(200 * (5000 / M))
    epoch_thresholds = [int(x) for x in [0.3 * epochs, 0.6 * epochs, 0.8 * epochs]]

    optimizer = optim.SGD(student_net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)

    best_test_set_accuracy = 0
    teacher_net.eval()

    for epoch in tqdm(range(epochs)):

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

        optimizer = adjust_learning_rate(optimizer, epoch + 1, epoch_thresholds=epoch_thresholds)

        if epoch >= epoch_thresholds[-1] and epoch % int((5000 / M)) == 0:

            epoch_accuracy = _test_set_eval(student_net, device, test_loader)

            if log:
                with open(os.path.join('./', logfile), "a") as temp:
                    temp.write('Accuracy at epoch {} is {}%\n'.format(epoch + 1, epoch_accuracy))

            if epoch_accuracy > best_test_set_accuracy:
                best_test_set_accuracy = epoch_accuracy
                if checkpoint:
                    torch.save(student_net.state_dict(), checkpointFile)

    if checkpoint:
        checkpoint_file_final = '{}-final-dict.pth'.format(checkpointFile.replace('-dict.pth', ''))
        torch.save(student_net.state_dict(), checkpoint_file_final)

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
        teacher_str = "WideResNet-{}-{}".format(wrn_depth_teacher, wrn_width_teacher)
        student_str = "WideResNet-{}-{}".format(wrn_depth_student, wrn_width_student)
        logfile = "_Teacher-{}-Student-{}-{}-M-{}.txt".format(teacher_str, student_str, kd_att_configurations.dataset, M)
        print(logfile)
        with open(os.path.join('./', logfile), "w") as temp:
            temp.write('KD_ATT with teacher {} and student {} in {} with M={}\n'.format(teacher_str, student_str,
                                                                                        kd_att_configurations.dataset,
                                                                                        M))
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

        teacher_net = WideResNet(d=wrn_depth_teacher, k=wrn_width_teacher, n_classes=10, input_features=3,
                                 output_features=16, strides=strides)
        teacher_net = teacher_net.to(device)
        if dataset.lower() == 'cifar10':
            torch_checkpoint = torch.load('./PreTrainedModels/PreTrainedScratches/CIFAR10/wrn-{}-{}-seed-{}-dict.pth'
                    .format(wrn_depth_teacher, wrn_width_teacher,seed),map_location=device)
        else:
            torch_checkpoint = torch.load(
                './PreTrainedModels/PreTrainedScratches/SVHN/wrn-{}-{}-seed-svhn-{}-dict.pth'
                    .format(wrn_depth_teacher,wrn_width_teacher, seed),map_location=device)

        teacher_net.load_state_dict(torch_checkpoint)

        student_net = WideResNet(d=wrn_depth_student, k=wrn_width_student, n_classes=10, input_features=3,
                                 output_features=16, strides=strides)
        student_net = student_net.to(device)

        checkpointFile = 'kd_att_teacher_wrn-{}-{}_student_wrn-{}-{}-M-{}-seed-{}-{}-dict.pth'.format(wrn_depth_teacher,
                                                                                                      wrn_width_teacher,
                                                                                                      wrn_depth_student,
                                                                                                      wrn_width_student,
                                                                                                      M, seed,
                                                                                                      dataset) if checkpoint else ''
        if M != 0:

            best_test_set_accuracy = _train_seed_kd_att(teacher_net, student_net, M, loaders, device, log, checkpoint,
                                                        logfile, checkpointFile)

            if log:
                with open(os.path.join('./', logfile), "a") as temp:
                    temp.write('Best test set accuracy of seed {} is {}\n'.format(seed, best_test_set_accuracy))

            test_set_accuracies.append(best_test_set_accuracy)

            if log:
                with open(os.path.join('./', logfile), "a") as temp:
                    temp.write('Best test set accuracy of seed {} is {}\n'.format(seed, best_test_set_accuracy))

        else:

            best_test_set_accuracy = _test_set_eval(student_net, device, test_loader)
            test_set_accuracies.append(best_test_set_accuracy)

    mean_test_set_accuracy, std_test_set_accuracy = np.mean(test_set_accuracies), np.std(test_set_accuracies)

    if log:
        with open(os.path.join('./', logfile), "a") as temp:
            temp.write(
                'Mean test set accuracy is {} with standard deviation equal to {}\n'.format(mean_test_set_accuracy,
                                                                                            std_test_set_accuracy))


if __name__ == '__main__':
    import argparse

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    parser = argparse.ArgumentParser(description='WideResNet Scratches')

    parser.add_argument('-config', '--config', help='Training Configurations', required=True)

    args = parser.parse_args()

    train(args)

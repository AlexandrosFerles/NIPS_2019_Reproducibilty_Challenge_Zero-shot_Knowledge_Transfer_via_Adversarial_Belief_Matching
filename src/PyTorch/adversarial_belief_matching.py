import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
from utils import json_file_to_pyobj, get_matching_indices
from WideResNet import WideResNet
from copy import deepcopy
from tqdm import tqdm


def _load_teacher_and_student(abm_configurations, seed, device):
    wrn_depth_teacher = abm_configurations.wrn_depth_teacher
    wrn_width_teacher = abm_configurations.wrn_width_teacher
    wrn_depth_student = abm_configurations.wrn_depth_student
    wrn_width_student = abm_configurations.wrn_width_student

    # KD-ATT or Zero-Shot for student
    mode = abm_configurations.mode

    dataset = abm_configurations.dataset.lower()

    strides = [1, 1, 2, 2]

    teacher_net = WideResNet(d=wrn_depth_teacher, k=wrn_width_teacher, n_classes=10, input_features=3,
                             output_features=16, strides=strides)
    teacher_net = teacher_net.to(device)

    if dataset.lower() == 'cifar10':
        torch_checkpoint = torch.load(
            './PreTrainedModels/PreTrainedScratches/CIFAR10/wrn-{}-{}-seed-{}-dict.pth'.format(wrn_depth_teacher,
                                                                                               wrn_width_teacher, seed),
            map_location=device)
    elif dataset.lower() == 'svhn':
        torch_checkpoint = torch.load(
            './PreTrainedModels/PreTrainedScratches/SVHN/wrn-{}-{}-seed-svhn-{}-dict.pth'.format(wrn_depth_teacher,
                                                                                                 wrn_width_teacher,
                                                                                                 seed),
            map_location=device)
    else:
        raise ValueError('Dataset not found')

    teacher_net.load_state_dict(torch_checkpoint)

    student_net = WideResNet(d=wrn_depth_student, k=wrn_width_student, n_classes=10, input_features=3,
                             output_features=16, strides=strides)
    student_net = student_net.to(device)

    if mode.lower() == 'kd-att':
        if dataset.lower() == 'cifar10':
            torch_checkpoint = torch.load(
                './PreTrainedModels/KD-ATT/CIFAR10/kd_att_teacher_wrn-{}-{}_student_wrn-{}-{}-M-0-seed-{}-CIFAR10-dict.pth'.format(
                    wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, seed),
                map_location=device)
        elif dataset.lower() == 'svhn':
            torch_checkpoint = torch.load(
                './PreTrainedModels/KD-ATT/SVHN/kd_att_teacher_wrn-{}-{}_student_wrn-{}-{}-M-0-seed-{}-SVHN-dict.pth'.format(
                    wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, seed),
                map_location=device)
        else:
            raise ValueError('Dataset not found')
    else:
        if dataset.lower() == 'cifar10':
            torch_checkpoint = torch.load(
                './PreTrainedModels/Zero-Shot/CIFAR10/reproducibility_zero_shot_teacher_wrn-{}-{}_student_wrn-{}-{}-M-0-seed-{}-CIFAR10-dict.pth'.format(
                    wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, seed),
                map_location=device)
        elif dataset.lower() == 'svhn':
            torch_checkpoint = torch.load(
                './PreTrainedModels/Zero-Shot/SVHN/reproducibility_zero_shot_teacher_wrn-{}-{}_student_wrn-{}-{}-M-0-seed-{}-SVHN-dict.pth'.format(
                    wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, seed),
                map_location=device)
        else:
            raise ValueError('Dataset not found')

    student_net.load_state_dict(torch_checkpoint)

    return teacher_net, student_net


def adversarial_belief_matching(args):
    json_options = json_file_to_pyobj(args.config)
    abm_configurations = json_options.abm_setting

    wrn_depth_teacher = abm_configurations.wrn_depth_teacher
    wrn_width_teacher = abm_configurations.wrn_width_teacher
    wrn_depth_student = abm_configurations.wrn_depth_student
    wrn_width_student = abm_configurations.wrn_width_student

    dataset = abm_configurations.dataset.lower()
    seeds = abm_configurations.seeds
    mode = abm_configurations.mode

    if torch.cuda.is_available():
        device = torch.device('cuda:3')
    else:
        device = torch.device('cpu')


    print(test_loader.__len__())
    for seed in seeds:

        teacher_net, student_net = _load_teacher_and_student(abm_configurations, seed, device)
        test_loader = get_matching_indices(dataset, teacher_net, student_net, device, n=1000)
        cnt = test_loader.__len__()

        teacher_net.eval()

        criterion = nn.CrossEntropyLoss()

        eta = 1
        K = 100

        student_image_average_transition_curves_acc, teacher_image_average_transition_curves_acc = [], []
        mean_transition_error = 0

        # count on how many test set samples teacher and student initially agree (and they are correct too!)
        cnt = 0
        for data in tqdm(test_loader):

            images, _ = data
            images = images.to(device)

            student_net.eval()
            student_outputs = student_net(images)[0]
            _, student_predicted = torch.max(student_outputs.data, 1)

            teacher_outputs = teacher_net(images)[0]
            _, teacher_predicted = torch.max(teacher_outputs.data, 1)

            if student_predicted == teacher_predicted:

                x0 = deepcopy(images.detach())
                student_transition_curves, teacher_transition_curves = [], []

                for fake_label in range(0, 10):

                    if fake_label != student_predicted:

                        fake_label = torch.Tensor([fake_label]).long().to(device)
                        student_probs_acc, teacher_probs_acc = [], []

                        x_adv = deepcopy(x0)
                        x_adv.requires_grad = True

                        for _ in range(K):
                            student_fake_outputs = student_net(x_adv)[0]
                            with torch.no_grad():
                                teacher_fake_outputs = teacher_net(x_adv)[0]
                            loss = criterion(student_fake_outputs, fake_label)

                            student_net.zero_grad()
                            loss.backward()
                            x_adv.data -= eta * x_adv.grad.data
                            x_adv.grad.data.zero_()

                            teacher_probs = F.softmax(teacher_fake_outputs, dim=1)
                            student_probs = F.softmax(student_fake_outputs, dim=1)

                            pj_b = teacher_probs[0][fake_label].item()
                            pj_a = student_probs[0][fake_label].item()

                            student_probs_acc.append(pj_a)
                            teacher_probs_acc.append(pj_b)

                            mean_transition_error += abs(pj_b- pj_a)

                        student_transition_curves.append(student_probs_acc)
                        teacher_transition_curves.append(teacher_probs_acc)

                else:
                    continue

            if student_predicted == teacher_predicted:

                student_image_average_transition_curves_acc.append(np.average(np.array(student_transition_curves), axis=0))
                teacher_image_average_transition_curves_acc.append(np.average(np.array(teacher_transition_curves), axis=0))

        student_image_average_transition_curves_acc_np = np.average(np.array(student_image_average_transition_curves_acc), axis=0)
        teacher_image_average_transition_curves_acc_np = np.average(np.array(teacher_image_average_transition_curves_acc), axis=0)

        with open('Teacher_WRN-{}-{}_transition_curve-{}-seed-{}.pickle'.format(wrn_depth_teacher, wrn_width_teacher,mode, seed),'wb') as teacher_pickle:
            pickle.dump(teacher_image_average_transition_curves_acc_np, teacher_pickle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('Student_WRN-{}-{}_transition_curve-{}-seed-{}.pickle'.format(wrn_depth_student, wrn_width_student, mode, seed), 'wb') as student_pickle:
            pickle.dump(student_image_average_transition_curves_acc_np, student_pickle, protocol=pickle.HIGHEST_PROTOCOL)

        # Average MTE over C-1 classes, K adversarial steps and correct initial samples
        mean_transition_error /= float(9 * K * cnt)
        with open('Teacher_WRN-{}-{}-Student_WRN-{}-{}-{}-MTE.txt'.format(wrn_depth_teacher,
                                                                          wrn_width_teacher,
                                                                          wrn_depth_student,
                                                                          wrn_width_student,
                                                                          mode), 'w') as logfile:
            logfile.write(
                    'Teacher WideResNet-{}-{} and Student WideResNet-{}-{} trained with {} Mean Transition Error on seed {}: {}\n'.format(
                        wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, mode, seed,
                        mean_transition_error))


if __name__ == '__main__':
    import argparse

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    parser = argparse.ArgumentParser(description='WideResNet Scratches')

    parser.add_argument('-config', '--config', help='Training Configurations', required=True)

    args = parser.parse_args()

    adversarial_belief_matching(args)

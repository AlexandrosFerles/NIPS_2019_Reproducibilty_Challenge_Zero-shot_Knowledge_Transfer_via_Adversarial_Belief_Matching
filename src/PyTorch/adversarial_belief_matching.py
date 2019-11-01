import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
from utils import json_file_to_pyobj
from WideResNet import WideResNet
from copy import deepcopy


def adversarial_belief_evaluation(args):
    
    json_options = json_file_to_pyobj(args.config)
    abm_configurations = json_options.abm_setting

    wrn_depth_teacher = abm_configurations.wrn_depth_teacher
    wrn_width_teacher = abm_configurations.wrn_width_teacher
    wrn_depth_student = abm_configurations.wrn_depth_student
    wrn_width_student = abm_configurations.wrn_width_student
    
    teacher_checkpoint_path = abm_configurations.teacher_checkpoint_path
    zero_shot_student_path = abm_configurations.zero_shot_student_path

    # KD-ATT or Zero-Shot for student
    mode = abm_configurations.mode

    dataset = abm_configurations.dataset.lower()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if dataset == 'cifar10':
        from utils import cifar10loaders
        _, test_loader = cifar10loaders(test_batch_size=1)
    elif dataset =='svhn':
        from utils import svhnLoaders
        _, test_loader = svhnLoaders(test_batch_size=1)

    strides = [1, 1, 2, 2]

    teacher_net = WideResNet(d=wrn_depth_teacher, k=wrn_width_teacher, n_classes=10, input_features=3, output_features=16, strides=strides)
    teacher_net = teacher_net.to(device)

    student_net = WideResNet(d=wrn_depth_student, k=wrn_width_student, n_classes=10, input_features=3, output_features=16, strides=strides)
    student_net = student_net.to(device)

    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    student_checkpoint = torch.load(zero_shot_student_path, map_location=device)

    teacher_net.load_state_dict(teacher_checkpoint)
    student_net.load_state_dict(student_checkpoint)

    teacher_net.eval()

    criterion = nn.CrossEntropyLoss()

    eta = 1
    K = 100

    student_image_average_transition_curves_acc, teacher_image_average_transition_curves_acc = [], []
    mean_transition_error = 0

    # count on how many test set samples teacher and student initially agree (and they are correct too!)
    cnt = 0
    for data in test_loader:

        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        student_net.eval()
        student_outputs = student_net(images)[0]
        _, student_predicted = torch.max(student_outputs.data, 1)

        teacher_outputs = teacher_net(images)[0]
        _, teacher_predicted = torch.max(teacher_outputs.data, 1)

        student_net.train()
        if student_predicted == teacher_predicted == labels:
            cnt+=1
            x0 = deepcopy(images.detach())
            student_transition_curves, teacher_transition_curves = [], []

            for fake_label in range(0, 10):

                student_probs_acc, teacher_probs_acc = [], []

                if fake_label != labels:
                    x_adv = deepcopy(x0.detach().requuires_grad())
                    optimizer = optim.SGD(x_adv, lr=eta)

                    for _ in range(K):
                        student_fake_outputs = student_net(x_adv)[0]
                        with torch.no_grad():
                            teacher_fake_outputs = teacher_net(x_adv)[0]

                        loss = criterion(student_fake_outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        teacher_probs = F.softmax(teacher_fake_outputs)
                        student_probs = F.softmax(teacher_fake_outputs)

                        teacher_probs_acc.append(teacher_probs[fake_label])
                        student_probs_acc.append(student_probs[fake_label])

                        mean_transition_error += abs(teacher_probs[fake_label].item() - student_probs[fake_label].item())

                    student_transition_curves.append(student_probs)
                    teacher_transition_curves.append(teacher_probs)
            else:
                continue

        student_image_average_transition_curves_acc.append(np.average(np.array(student_transition_curves), axis=0))
        teacher_image_average_transition_curves_acc.append(np.average(np.array(teacher_transition_curves), axis=0))

    student_image_average_transition_curves_acc_np = np.average(np.array(student_image_average_transition_curves_acc), axis=0)
    teacher_image_average_transition_curves_acc_np = np.average(np.array(teacher_image_average_transition_curves_acc), axis=0)

    with open('Teacher_WRN-{}-{}_transition_curve-{}.pickle'.format(wrn_depth_teacher, wrn_width_teacher, mode), 'wb') as teacher_pickle:
        pickle.dump(teacher_image_average_transition_curves_acc_np, teacher_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Student_WRN-{}-{}_transition_curve-{}.pickle'.format(wrn_depth_student, wrn_width_student, mode), 'wb') as student_pickle:
        pickle.dump(student_image_average_transition_curves_acc_np, student_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    # Average MTE over C-1 classes, K adversarial steps and correct initial samples
    mean_transition_error /= float(9*K*cnt)
    with open('Teacher_WRN-{}-{}-Student_WRN-{}-{}-{}'.format(wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, mode), 'w') as logfile:
        logfile.write('Teacher WideResNet-{}-{} and Student WideResNet-{}-{} trained with {} Mean Transition Error: {}'.format(wrn_depth_teacher, wrn_width_teacher, wrn_depth_student, wrn_width_student, mean_transition_error))


if __name__ == '__main__':
    import argparse

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    parser = argparse.ArgumentParser(description='WideResNet Scratches')

    parser.add_argument('-config', '--config', help='Training Configurations', required=True)

    args = parser.parse_args()

    adversarial_belief_evaluation(args)
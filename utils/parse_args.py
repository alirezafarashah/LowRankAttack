import argparse
import copy
import logging


def get_args():
    parser = argparse.ArgumentParser()

    # Architecture settings
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='One of: CIFAR10, CIFAR100 or SVHN')
    parser.add_argument('--architecture', default='PreActResNet18', type=str,
                        help='One of: resnet18, resnet18. Default: preactresnet18.')

    # Training schedule settings
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/path/to/datasets/', type=str)

    # Adversarial training and evaluation settings
    parser.add_argument('--epsilon', default=128, type=int)
    parser.add_argument('--pgd-alpha', default=15, type=int)
    # Config paths
    parser.add_argument('--model-path', default='/kaggle/working/',
                        type=str, help='Pretrained model path')
    parser.add_argument('--save-path', default='/kaggle/working/',
                        type=str, help='Path to save the tensor V')
    parser.add_argument('--log-dir', default='/kaggle/working/',
                        type=str, help='Path to save the tensor V')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    # Optimization args
    parser.add_argument('--attack-iters', default=20, type=int, help='Number of steps to optimize Ui')
    return parser.parse_args()


def get_eval_args():
    parser = argparse.ArgumentParser()

    # Architecture settings
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='One of: CIFAR10, CIFAR100 or SVHN')
    parser.add_argument('--architecture', default='PreActResNet18', type=str,
                        help='One of: resnet18, resnet50, preactresnet18. Default: preactresnet18.')

    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/path/to/datasets/', type=str)

    # Config paths
    parser.add_argument('--model-path', default='/kaggle/working/',
                        type=str, help='Pretrained model path')
    parser.add_argument('--perturbations-path', default='/kaggle/working/perturbations.pt',
                        type=str, help='Path of perturbations')

    parser.add_argument('--log-dir', default='/kaggle/working/',
                        type=str, help='Path to save the tensor V')

    return parser.parse_args()

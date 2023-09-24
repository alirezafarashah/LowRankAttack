import argparse
import copy
import logging
from argparse import ArgumentParser


def get_train_args():
    parser: ArgumentParser = argparse.ArgumentParser()

    # Architecture settings
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='One of: CIFAR10, CIFAR100 or SVHN')
    parser.add_argument('--architecture', default='PreActResNet18', type=str,
                        help='One of: resnet18, resnet50, preactresnet18. Default: preactresnet18.')
    # Training schedule settings
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/path/to/datasets/', type=str)
    parser.add_argument('--epochs', default=3, type=int)

    # Adversarial training and evaluation settings
    parser.add_argument('--epsilon', default=3, type=float)
    parser.add_argument('--validation', action='store_true',
                        help='Validate attack')

    # Config paths
    parser.add_argument('--model-path', default='/kaggle/working/',
                        type=str, help='Pretrained model path')
    parser.add_argument('--save-path', default='/kaggle/working/',
                        type=str, help='Path to save the tensor V')
    parser.add_argument('--log-dir', default='/kaggle/working/',
                        type=str, help='Path to save the tensor V')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    # Optimization args
    parser.add_argument('--inner-steps', default=100, type=int, help='Number of steps to optimize Ui')
    parser.add_argument('--u-rate', default=1e-1, type=float, help='Learning rate for Ui optimization')
    parser.add_argument('--v-rate', default=1e-2, type=float, help='Learning rate for V optimization')
    parser.add_argument('--max-fro', default=3, type=float, help='Maximum frobenius norm of V')
    parser.add_argument('--v-dim', default=100, type=int, help='Dimension of V')
    parser.add_argument('--init-v', default=0.05, type=float, help='initial value of V')
    return parser.parse_args()


def get_eval_args():
    parser: ArgumentParser = argparse.ArgumentParser()

    # Architecture settings
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='One of: CIFAR10, CIFAR100 or SVHN')
    parser.add_argument('--architecture', default='PreActResNet18', type=str,
                        help='One of: resnet18, resnet50, preactresnet18. Default: preactresnet18.')

    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/path/to/datasets/', type=str)

    # Config paths
    parser.add_argument('--model-path', default='/kaggle/working/',
                        type=str, help='Pretrained model path')
    parser.add_argument('--U-path', default='None',
                        type=str, help='Path of trained U tensor')
    parser.add_argument('--V-path', default='/kaggle/working/V.pt',
                        type=str, help='Path of trained V tensor')

    parser.add_argument('--log-dir', default='/kaggle/working/',
                        type=str, help='Path to save the tensor V')

    parser.add_argument('--epsilon', default=2.0, type=float)
    parser.add_argument('--u-rate', default=1e-1, type=float, help='Learning rate for Ui optimization')
    parser.add_argument('--inner-steps', default=50, type=int, help='Number of steps to optimize Ui')

    return parser.parse_args()

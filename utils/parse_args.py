import argparse
import copy
import logging


def get_args():
    parser = argparse.ArgumentParser()

    # Architecture settings
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='One of: CIFAR10, CIFAR100 or SVHN')
    parser.add_argument('--architecture', default='PreActResNet18', type=str,
                        help='One of: wideresnet, preactresnet18. Default: preactresnet18.')
    parser.add_argument('--wide_resnet_depth', default=28, type=int, help='WideResNet depth')
    parser.add_argument('--wide_resnet_width', default=10, type=int, help='WideResNet width')
    parser.add_argument('--wide_resnet_dropout_rate', default=0.3, type=float, help='WideResNet dropout rate')

    # Training schedule settings
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/path/to/datasets/', type=str)
    parser.add_argument('--epochs', default=10, type=int)

    # Adversarial training and evaluation settings
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--validation', action='store_true',
                        help='Validate attack')

    # Config paths
    parser.add_argument('--model-path', default='/kaggle/working/',
                        type=str, help='Pretrained model path')
    parser.add_argument('--save-path', default='/kaggle/working/',
                        type=str, help='Path to save the tensor V')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    # Optimization args
    parser.add_argument('--inner-steps', default=10, type=int, help='Number of steps to optimize Ui')
    parser.add_argument('--u-rate', default=1e-3, type=float, help='Learning rate for Ui optimization')
    parser.add_argument('--v-rate', default=1e-3, type=float, help='Learning rate for V optimization')
    parser.add_argument('--lambda-1', default=1e-3, type=float, help='lambda-1 parameter in loss')
    parser.add_argument('--lambda-2', default=1e-3, type=float, help='lambda-2 parameter in loss')

    return parser.parse_args()

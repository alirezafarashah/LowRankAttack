import os
import time
from tqdm import tqdm
import logging

import numpy as np
import torch
import torch.nn.functional as F

from architectures.preact_resnet import PreActResNet18
from architectures.wide_resnet import Wide_ResNet

from utils.data_utils import CIFAR10Utils, CIFAR100Utils
from utils.attack_utils import *
from utils.parse_args import get_args

CHANNELS = 3
logger = logging.getLogger(__name__)


def test():
    args = get_args()
    print(args)
    print('Defining data object')
    if args.dataset.upper() == 'CIFAR10':
        data_utils = CIFAR10Utils()
    elif args.dataset.upper() == 'CIFAR100':
        data_utils = CIFAR100Utils()
    else:
        raise ValueError('Unsupported dataset.')

    attack_utils = AttackUtils(data_utils.lower_limit, data_utils.upper_limit, data_utils.std)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=args.log_dir + 'output.log')
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    valid_size = 0
    (train_loader, test_loader, robust_test_loader,
     valid_loader, train_idx, valid_idx) = data_utils.get_indexed_loaders(args.data_dir,
                                                                          args.batch_size,
                                                                          valid_size=valid_size)

    # Define architecture
    args.num_classes = data_utils.max_label + 1  # Labels start from 0
    if args.architecture.upper() == 'PREACTRESNET18':
        model = PreActResNet18(num_classes=args.num_classes).cuda()
    elif args.architecture.upper() in 'WIDERESNET':
        model = Wide_ResNet(args.wide_resnet_depth,
                            args.wide_resnet_width,
                            args.wide_resnet_dropout_rate,
                            num_classes=args.num_classes).cuda()
    else:
        raise ValueError('Unknown architecture.')

    model_path = args.model_path
    if not os.path.exists(model_path):
        raise ValueError('Pretrained model does not exist.')

    model.load_state_dict(torch.load(model_path))
    logger.info("Pretrained model loaded successfully.")
    print("Pretrained model loaded successfully.")
    model.eval()
    test_loss, test_acc = evaluate_model(model, test_loader)
    logger.info(f"Evaluate model on clean dataset, test loss: {test_loss}, test acc: {test_acc}")
    print(f"Evaluate model on clean dataset, test loss: {test_loss}, test acc: {test_acc}")

    attack_iters = args.attack_iters
    # Evaluation final tensor
    logger.info("Training finished, starting evaluation.")
    print('Training finished, starting evaluation.')
    pgd_alpha = args.pgd_alpha / 255.
    epsilon = args.epsilon / 255.
    test_loss, test_acc, perturbations = attack_utils.evaluate_pgd(train_loader, model, attack_iters, 1, epsilon,
                                                                   pgd_alpha)
    torch.save(perturbations, args.save_path + "perturbations.pt")
    logger.info(f"test loss: {test_loss}, test acc: {test_acc}")
    print(f"test loss: {test_loss}, test acc: {test_acc}")
    logger.info('Finished evaluating final tensor.')
    print('Finished evaluating final tensor.')


if __name__ == "__main__":
    test()

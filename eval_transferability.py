import os
import time
from tqdm import tqdm
import logging

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from architectures.preact_resnet import PreActResNet18
from architectures.wide_resnet import Wide_ResNet
from architectures.resnet import ResNet50, ResNet18

from utils.data_utils import CIFAR10Utils, CIFAR100Utils
from utils.attack_utils import *
from utils.parse_args import get_eval_args

logger = logging.getLogger(__name__)


def eval_transferability():
    args = get_eval_args()
    print(args)
    if args.dataset.upper() == 'CIFAR10':
        data_utils = CIFAR10Utils()
    elif args.dataset.upper() == 'CIFAR100':
        data_utils = CIFAR100Utils()
    else:
        raise ValueError('Unsupported dataset.')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=args.log_dir + 'output.log')
    logger.info(args)


    eval_dataset = data_utils.get_eval_dataset(args.data_dir)
    eval_loader = data_utils.get_indexed_loaders(args.data_dir, args.batch_size, valid_size=0)[1]

    # Define architecture
    args.num_classes = data_utils.max_label + 1  # Labels start from 0
    if args.architecture.upper() == 'PREACTRESNET18':
        model = PreActResNet18(num_classes=args.num_classes).cuda()
    elif args.architecture.upper() in 'RESNET18':
        model = ResNet18(num_classes=args.num_classes).cuda()
    elif args.architecture.upper() in 'RESNET50':
        model = ResNet50(num_classes=args.num_classes).cuda()

    else:
        raise ValueError('Unknown architecture.')

    model_path = args.model_path
    if not os.path.exists(model_path):
        raise ValueError('Pretrained model does not exist.')
    model.load_state_dict(torch.load(model_path))
    logger.info("Pretrained model loaded successfully.")
    print("Pretrained model loaded successfully.")
    model.eval()
    test_loss, test_acc = evaluate_model(model, eval_loader)
    logger.info(f"Evaluate model on clean dataset, test loss: {test_loss}, test acc: {test_acc}")
    print(f"Evaluate model on clean dataset, test loss: {test_loss}, test acc: {test_acc}")

    logger.info("starting evaluation.")
    print('starting evaluation.')
    U = torch.load(args.U_path)
    V = torch.load(args.V_path)
    test_loss = 0
    test_acc = 0
    n = 0
    with torch.no_grad():
        for Ui, batch_idx in U:
            my_subset = Subset(eval_dataset, batch_idx)
            loader = DataLoader(my_subset, batch_size=128)
            X, y, idx = next(iter(loader))
            X, y = X.cuda(), y.cuda()
            output = model(X + torch.matmul(Ui, V).reshape(X.shape))
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        test_loss = test_loss / n
        test_acc = test_acc / n
        print(f"test loss on U and V: {test_loss}, test acc: {test_acc}")
        logger.info(f"test loss on U and V: {test_loss}, test acc: {test_acc}")


if __name__ == "__main__":
    eval_transferability()

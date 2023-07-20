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


def train():
    global U, V
    args = get_args()
    print(args)

    print('Defining data object')
    if args.dataset.upper() == 'CIFAR10':
        data_utils = CIFAR10Utils()
    elif args.dataset.upper() == 'CIFAR100':
        data_utils = CIFAR100Utils()
    else:
        raise ValueError('Unsupported dataset.')

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename='output.log')
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

    inner_steps = args.inner_steps  # > 100
    lambda_1 = args.lambda_1
    u_rate = args.u_rate  # < 1/(2 * lambda)
    v_rate = args.v_rate  # < 1/(2 * lambda)
    d = data_utils.img_size[0] * data_utils.img_size[1] * CHANNELS
    max_norm = args.epsilon / 255.
    V = torch.rand(100, d).cuda()
    V = clamp_operator_norm(V)
    print("fro norm of V: ", torch.pow(torch.norm(V, p='fro'), 2))
    start_train_time = time.time()
    logger.info('Epoch \t Seconds')
    print('Epoch \t Seconds')

    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        U = []
        for i, (X, y, batch_idx) in enumerate(tqdm(train_loader)):
            X, y = X.cuda(), y.cuda()
            Ui = (2 * max_norm * torch.rand(X.shape[0], 100) - max_norm).cuda()
            # Ui optimization step
            V.requires_grad = False
            for j in range(inner_steps):
                Ui.requires_grad = True
                output = model(X + torch.matmul(Ui, V).reshape(X.shape))
                reg_term1 = torch.sum(torch.pow(torch.norm(Ui, p=2, dim=1), 2))
                loss = F.cross_entropy(output, y) - lambda_1 * reg_term1
                grad = torch.autograd.grad(loss, Ui)[0]
                grad = grad.detach()
                Ui = Ui + u_rate * grad
                Ui = Ui.detach()

            # V optimization step
            V.requires_grad = True
            Ui.requires_grad = False
            output = model(X + torch.matmul(Ui, V).reshape(X.shape))
            loss = F.cross_entropy(output, y)
            grad = torch.autograd.grad(loss, V)[0]
            grad = grad.detach()
            V = V + v_rate * torch.sign(grad)
            V = clamp_operator_norm(V)
            V = V.detach()
            Ui = Ui.detach()
            U.append(Ui)

        if args.validation:
            test_loss, test_acc = evaluate_low_rank(model, V, U, train_loader)
            logger.info(f"test loss: {test_loss}, test acc: {test_acc}")
            print(f"test loss: {test_loss}, test acc: {test_acc}")
            logger.info("l2 norm of Ui: %.4f", torch.sum(torch.pow(torch.norm(Ui, p=2, dim=1), 2)).item())
            print("l2 norm of Ui: ", torch.sum(torch.pow(torch.norm(Ui, p=2, dim=1), 2)))
            logger.info("fro norm of V: %.4f", torch.pow(torch.norm(V, p='fro'), 2).item())
            print("fro norm of V: ", torch.pow(torch.norm(V, p='fro'), 2))

        epoch_time = time.time()
        print(epoch, epoch_time - start_epoch_time)

    train_time = time.time()
    torch.save(V, args.save_path + "V.pt")
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)
    print('Total train time: %.4f minutes', (train_time - start_train_time) / 60)

    # Evaluation final tensor
    logger.info("Training finished, starting evaluation.")
    print('Training finished, starting evaluation.')
    test_loss, test_acc = evaluate_low_rank(model, V, U, train_loader)
    logger.info(f"test loss: {test_loss}, test acc: {test_acc}")
    print(f"test loss: {test_loss}, test acc: {test_acc}")
    logger.info('Finished evaluating final tensor.')
    print('Finished evaluating final tensor.')


if __name__ == "__main__":
    train()

import os
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from architectures.preact_resnet import PreActResNet18
from architectures.wide_resnet import Wide_ResNet

from utils.data_utils import CIFAR10Utils, CIFAR100Utils
from utils.attack_utils import AttackUtils
from utils.parse_args import get_args

CHANNELS = 3


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

    print('Defining attack object')
    attack_utils = AttackUtils(data_utils.lower_limit, data_utils.upper_limit, data_utils.std)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    valid_size = 0
    (train_loader, test_loader, robust_test_loader,
     valid_loader, train_idx, valid_idx) = data_utils.get_indexed_loaders(args.data_dir,
                                                                          args.batch_size,
                                                                          valid_size=valid_size)
    # Adv training and test settings
    epsilon = (args.epsilon / 255.) / data_utils.std
    inner_steps = args.inner_steps

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
    print("Pretrained model loaded successfully.")
    model.eval()
    test_loss, test_acc = attack_utils.evaluate_model(model, test_loader)
    print(f"Evaluate model on clean dataset, test loss: {test_loss}, test acc: {test_acc}")

    lambda_1 = args.lambda_1
    lambda_2 = args.lambda_2
    u_rate = args.u_rate
    v_rate = args.v_rate
    d = 100 * data_utils.img_size[1] * CHANNELS
    V = torch.rand(d, d).cuda()

    start_train_time = time.time()
    print('Epoch \t Seconds')
    iter_count = 0
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        U = []
        for i, (X, y, batch_idx) in enumerate(tqdm(train_loader)):
            X, y = X.cuda(), y.cuda()
            print(X.shape)
            Ui = torch.zeros_like(X.shape[0], X.shape[1], X.shape[2], 100).cuda()
            for j in range(len(epsilon)):
                Ui[:, j, :, :].uniform_(-epsilon[j][0][0].item(),
                                        epsilon[j][0][0].item())

            # Ui optimization step
            V.requires_grad = False
            for j in range(inner_steps):
                Ui.requires_grad = True
                output = model(X + torch.matmul(Ui.reshape(X.shape[0], -1), V).reshape(X.shape))
                reg_term1 = lambda_1 * torch.sum(torch.norm(Ui.reshape(X.shape[0], -1), p=2, dim=1))
                loss = F.cross_entropy(output, y) - reg_term1
                grad = torch.autograd.grad(loss, Ui)[0]
                grad = grad.detach()
                Ui = Ui + u_rate * grad
                Ui = Ui.detach()

            # V optimization step
            V.requires_grad = True
            Ui.requires_grad = False
            output = model(X + torch.matmul(Ui.reshape(X.shape[0], -1), V).reshape(X.shape))
            reg_term1 = lambda_1 * torch.sum(torch.norm(Ui.reshape(X.shape[0], -1), p=2, dim=1))
            reg_term2 = lambda_2 * torch.norm(V, p='fro')
            loss = F.cross_entropy(output, y) - reg_term1 - reg_term2
            grad = torch.autograd.grad(loss, V)[0]
            grad = grad.detach()
            print(grad.shape)
            V = V + v_rate * grad
            Ui = Ui.detach()
            V = V.detach()
            U.append(Ui)

            iter_count += 1

        if args.validation:
            test_loss, test_acc = attack_utils.evaluate_low_rank(model, V, U, train_loader)
            print(f"test loss: {test_loss}, test acc: {test_acc}")

        epoch_time = time.time()
        print(epoch, epoch_time - start_epoch_time)

    train_time = time.time()
    torch.save(V, args.save_path + "V.pt")

    print('Total train time: %.4f minutes', (train_time - start_train_time) / 60)

    # Evaluation final tensor
    print('Training finished, starting evaluation.')
    test_loss, test_acc = attack_utils.evaluate_low_rank(model, V, U, train_loader)
    print(f"test loss: {test_loss}, test acc: {test_acc}")
    print('Finished evaluating final tensor.')


if __name__ == "__main__":
    train()

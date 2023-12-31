import os
import time
from tqdm import tqdm
import logging

import numpy as np
import torch
import torch.nn.functional as F

from architectures.preact_resnet import PreActResNet18
from architectures.wide_resnet import Wide_ResNet
from architectures.models import ResNet20, ResNet56, VGG16
from architectures.resnet import ResNet50, ResNet18

from utils.data_utils import CIFAR10Utils, CIFAR100Utils
from utils.attack_utils import *
from utils.parse_args import get_train_args

CHANNELS = 3
logger = logging.getLogger(__name__)


def train():
    global U, V
    args = get_train_args()
    print(args)

    print('Defining data object')
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
    elif args.architecture.upper() in 'RESNET18':
        model = ResNet18(num_classes=args.num_classes).cuda()
    elif args.architecture.upper() in 'RESNET20':
        model = ResNet20().cuda()
    elif args.architecture.upper() in 'RESNET50':
        model = ResNet50(num_classes=args.num_classes).cuda()
    elif args.architecture.upper() in 'RESNET56':
        model = ResNet56().cuda()
    elif args.architecture.upper() in 'VGG16':
        model = VGG16().cuda()
    else:
        raise ValueError('Unknown architecture.')

    model_path = args.model_path
    if args.architecture.upper() not in ['VGG16', 'RESNET20', 'RESNET56']:
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
    u_rate = args.u_rate
    v_rate = args.v_rate
    d = data_utils.img_size[0] * data_utils.img_size[1] * CHANNELS
    epsilon = args.epsilon
    V = torch.zeros(args.v_dim, d).cuda()
    V.uniform_(-args.init_v, args.init_v)
    V = fro_projection(V, args.max_fro)
    print("fro norm of V: ", torch.pow(torch.norm(V, p='fro'), 2))
    start_train_time = time.time()
    logger.info('Epoch \t Seconds')
    print('Epoch \t Seconds')
    final_U = []
    for epoch in range(args.epochs):
        U = []
        data = []
        start_epoch_time = time.time()
        for i, (X, y, batch_idx) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            Ui = torch.zeros(X.shape[0], args.v_dim).cuda()
            Ui.uniform_(-args.init_u, args.init_u)
            Ui = l2_projection(Ui, V.detach().clone(), epsilon)
            test_loss, test_acc = evaluate_batch(model, V.detach().clone(), Ui.detach().clone(), X, y)
            # print(f"1. test loss before train Ui: {test_loss}, test acc: {test_acc}")
            # logger.info(f"1. test loss before train Ui: {test_loss}, test acc: {test_acc}")

            # Ui optimization step
            V.requires_grad = False
            V_copy = V.detach().clone()
            for j in range(inner_steps):
                Ui.requires_grad = True
                output = model(X + torch.matmul(Ui, V_copy).reshape(X.shape))
                loss = F.cross_entropy(output, y)
                grad = torch.autograd.grad(loss, Ui)[0].detach()
                Ui = Ui.detach()
                Ui = Ui + u_rate * torch.div(grad, torch.linalg.vector_norm(grad, dim=1).unsqueeze(1))
                # Project onto l2 ball
                Ui = l2_projection(Ui, V_copy, epsilon)
                Ui = Ui.detach()

            test_loss, test_acc = evaluate_batch(model, V_copy, Ui.detach().clone(), X, y)
            # print(f"2. test loss after train Ui and before train V: {test_loss}, test acc: {test_acc}")
            # logger.info(f"2. test loss after train Ui and before train V: {test_loss}, test acc: {test_acc}")

            # V optimization step
            V.requires_grad = True
            Ui_copy = Ui.detach().clone()
            output = model(X + torch.matmul(Ui_copy, V).reshape(X.shape))
            loss = F.cross_entropy(output, y)
            grad = torch.autograd.grad(loss, V)[0].detach()
            V = V.detach()
            V = V + v_rate * torch.div(grad, torch.linalg.vector_norm(grad, dim=1).unsqueeze(1))
            V = fro_projection(V, args.max_fro)

            U.append(Ui.detach().clone())
            data.append((X.to(torch.device("cpu")), y.to(torch.device("cpu"))))
            V = V.detach()
            Ui = Ui.detach()
            # print_norms(model, V.detach().clone(), Ui.detach().clone(), X, y)
            if args.validation and (i + 1) % 50 == 0:
                v_rate = 0.9 * v_rate
                validation(model, V.detach().clone(), U, Ui.detach().clone(), data)

        epoch_time = time.time()
        print(epoch, epoch_time - start_epoch_time)

    train_time = time.time()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    torch.save(V, args.save_path + "V.pt")

    print("start training U for test set")
    logger.info("start training U for test set")
    U = []
    data = []
    for i, (X, y, batch_idx) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        Ui = torch.zeros(X.shape[0], args.v_dim).cuda()
        Ui.uniform_(-args.init_u, args.init_u)
        Ui = l2_projection(Ui, V.detach().clone(), epsilon)
        test_loss, test_acc = evaluate_batch(model, V.detach().clone(), Ui.detach().clone(), X, y)
        # print(f"1. test loss before train Ui: {test_loss}, test acc: {test_acc}")
        # logger.info(f"1. test loss before train Ui: {test_loss}, test acc: {test_acc}")

        # Ui optimization step
        V.requires_grad = False
        V_copy = V.detach().clone()
        for j in range(inner_steps):
            Ui.requires_grad = True
            output = model(X + torch.matmul(Ui, V_copy).reshape(X.shape))
            loss = F.cross_entropy(output, y)
            grad = torch.autograd.grad(loss, Ui)[0].detach()
            Ui = Ui.detach()
            Ui = Ui + u_rate * torch.div(grad, torch.linalg.vector_norm(grad, dim=1).unsqueeze(1))
            # Project onto l2 ball
            Ui = l2_projection(Ui, V_copy, epsilon)
            Ui = Ui.detach()
        test_loss, test_acc = evaluate_batch(model, V_copy, Ui.detach().clone(), X, y)
        # print(f"2. test loss after train Ui and before train V: {test_loss}, test acc: {test_acc}")
        # logger.info(f"2. test loss after train Ui and before train V: {test_loss}, test acc: {test_acc}")
        final_U.append((Ui.detach().clone(), batch_idx))
        U.append(Ui.detach().clone())
        data.append((X.to(torch.device("cpu")), y.to(torch.device("cpu"))))

    torch.save(final_U, args.save_path + "U.pt")
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)
    print('Total train time: %.4f minutes', (train_time - start_train_time) / 60)

    # Evaluation final tensor
    eval_attack(model, V, U, data)


def eval_attack(model, V, U, data):
    logger.info("Training finished, starting evaluation.")
    print('Training finished, starting evaluation.')
    test_loss, test_acc = evaluate_low_rank(model, V, U, data)
    logger.info(f"test loss: {test_loss}, test acc: {test_acc}")
    print(f"test loss: {test_loss}, test acc: {test_acc}")
    logger.info('Finished evaluating final tensor.')
    print('Finished evaluating final tensor.')


def print_norms(model, V, Ui, X, y):
    test_loss, test_acc = evaluate_batch(model, V, Ui, X, y)
    print(f"3. test loss after train V: {test_loss}, test acc: {test_acc}")
    logger.info(f"3. test loss after train V: {test_loss}, test acc: {test_acc}")
    print("4. l2 norm of Ui: ", torch.pow(torch.linalg.vector_norm(Ui), 2))
    logger.info("4. l2 norm of Ui: %.4f", torch.pow(torch.linalg.vector_norm(Ui), 2).item())
    print("5. l2 norm of UiV: ",
          torch.pow(torch.linalg.vector_norm(torch.matmul(Ui, V)), 2))
    logger.info("5. l2 norm of UiV: %.4f",
                torch.pow(torch.linalg.vector_norm(torch.matmul(Ui, V)),
                          2).item())
    print("6. fro norm of V: ", torch.pow(torch.linalg.matrix_norm(V), 2))
    logger.info("6. fro norm of V: %.4f", torch.pow(torch.linalg.matrix_norm(V), 2).item())
    print("7. nuclear norm of V: ", torch.linalg.matrix_norm(V, ord='nuc'))
    logger.info("7. nuclear norm of V: %.4f", torch.linalg.matrix_norm(V, ord='nuc').item())


def validation(model, V, U, Ui, data):
    print("test after 20 steps:")
    logger.info("test after 20 steps:")
    test_loss, test_acc = evaluate_low_rank(model, V, U, data)
    logger.info(f"test loss: {test_loss}, test acc: {test_acc}")
    print(f"test loss: {test_loss}, test acc: {test_acc}")
    Ui_norm_2 = torch.pow(torch.linalg.vector_norm(Ui), 2)
    V_norm_f = torch.pow(torch.linalg.matrix_norm(V), 2)
    logger.info("l2 norm of Ui: %.4f", Ui_norm_2.item())
    print("l2 norm of Ui: ", Ui_norm_2)
    logger.info("fro norm of V: %.4f", V_norm_f.item())
    print("fro norm of V: ", V_norm_f)


if __name__ == "__main__":
    train()

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.data import Dataset


class AttackUtils(object):

    def __init__(self, lower_limit, upper_limit, std):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.std = std

    def clamp(self, X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    def attack_pgd_l2(self, model, X, y, epsilon, alpha, attack_iters, restarts):
        max_loss = torch.zeros(y.shape[0]).cuda()
        max_delta = torch.zeros_like(X).cuda()
        for zz in range(restarts):
            delta = torch.zeros_like(X).cuda()
            delta.uniform_(-epsilon / 255., epsilon / 255.)
            delta.data = self.clamp(delta, self.lower_limit - X, self.upper_limit - X)
            delta.requires_grad = True
            for kk in range(attack_iters):
                output = model(X + delta)
                index = torch.where(output.max(1)[1] == y)
                if len(index[0]) == 0:
                    break
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                d = delta[index[0], :, :, :]
                g = grad[index[0], :, :, :]
                d = d + alpha * torch.div(g, torch.linalg.vector_norm(g, dim=1).unsqueeze(1))
                d = l2_project(d, epsilon)
                d = self.clamp(d, self.lower_limit - X[index[0], :, :, :], self.upper_limit - X[index[0], :, :, :])
                delta.data[index[0], :, :, :] = d
                delta.grad.zero_()
            all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)
        return max_delta

    def evaluate_pgd(self, test_loader, model, attack_iters, restarts=1, epsilon=2):
        alpha = epsilon / attack_iters * 2
        pgd_loss = 0
        pgd_acc = 0
        n = 0
        model.eval()
        for i, (X, y, batch_idx) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            pgd_delta = self.attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts)
            with torch.no_grad():
                output = model(X + pgd_delta)
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
        return pgd_loss / n, pgd_acc / n


def evaluate_model(model, test_loader):
    test_loss = 0
    test_acc = 0
    n = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n


def l2_project(X, e):
    X_norm = torch.linalg.vector_norm(X.reshape(-1), dim=1)
    epsilon_tensor = torch.full((X.reshape(-1).shape[0],), e).cuda()
    ones = torch.ones_like(epsilon_tensor).cuda()
    factor = torch.max(ones, X_norm / epsilon_tensor).unsqueeze(1)
    return torch.div(X.reshape(-1), factor).view(X.shape)

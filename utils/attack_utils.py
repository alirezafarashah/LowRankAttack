import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.data import Dataset

from tqdm import tqdm


class AttackUtils(object):

    def __init__(self, lower_limit, upper_limit, std):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.std = std

    def clamp(self, X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    def normalize(self, X, mu, std):
        return (X - mu) / std

    def attack_pgd(self, model, X, y, epsilon, alpha, attack_iters, restarts, ):
        max_loss = torch.zeros(y.shape[0]).cuda()
        max_delta = torch.zeros_like(X).cuda()
        for _ in range(restarts):
            delta = torch.zeros_like(X).cuda()
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
            delta = self.clamp(delta, self.lower_limit - X, self.upper_limit - X)
            delta.requires_grad = True
            for _ in range(attack_iters):
                output = model(X + delta)
                index = slice(None, None, None)
                if not isinstance(index, slice) and len(index) == 0:
                    break
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                d = delta[index, :, :, :]
                g = grad[index, :, :, :]
                x = X[index, :, :, :]
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
                d = self.clamp(d, self.lower_limit - x, self.upper_limit - x)
                delta.data[index, :, :, :] = d
                delta.grad.zero_()

            all_loss = F.cross_entropy(model(X + delta), y, reduction='none')
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)
        return max_delta

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
        mean_norm = 0
        step = 0
        perturbations = []
        for i, (X, y, batch_idx) in enumerate(tqdm(test_loader)):
            X, y = X.cuda(), y.cuda()
            pgd_delta = self.attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
            perturbations.append((pgd_delta.detach().clone(), batch_idx))
            with torch.no_grad():
                output = model(X + pgd_delta)
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
            mean_norm += torch.pow(torch.linalg.vector_norm(pgd_delta), 2)
            step += 1
        print("mean of l2 norm of delta: ", mean_norm / step)
        return pgd_loss / n, pgd_acc / n, perturbations


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
    shape = X.shape
    X = torch.flatten(X, start_dim=1)
    X_norm = torch.linalg.vector_norm(X, dim=1)
    epsilon_tensor = torch.full((X.shape[0],), e).cuda()
    ones = torch.ones_like(epsilon_tensor).cuda()
    factor = torch.max(ones, X_norm / epsilon_tensor).unsqueeze(1)
    return torch.div(X, factor).view(shape)

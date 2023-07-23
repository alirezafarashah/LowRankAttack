import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def clamp_operator_norm(V):
    V_array = V.detach().cpu().numpy()
    return torch.div(V, max(1, np.linalg.norm(V_array, 2)))


def clamp(self, X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def evaluate_low_rank(model, V, U, test_loader):
    test_loss = 0
    test_acc = 0
    n = 0
    with torch.no_grad():
        for i, (X, y, batch_idx) in enumerate(tqdm(test_loader)):
            X, y = X.cuda(), y.cuda()
            output = model(X + torch.matmul(U[batch_idx], V).reshape(X.shape))
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n


def evaluate_batch(model, V, Ui, X, y):
    test_loss = 0
    test_acc = 0
    n = 0
    with torch.no_grad():
        X, y = X.cuda(), y.cuda()
        output = model(X + torch.matmul(Ui, V).reshape(X.shape))
        loss = F.cross_entropy(output, y)
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)
    return test_loss / n, test_acc / n


def evaluate_model(model, test_loader):
    test_loss = 0
    test_acc = 0
    n = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(test_loader)):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n

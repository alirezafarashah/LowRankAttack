import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def fro_projection(V, d):
    V_array = V.detach().cpu().numpy()
    return torch.div(V, max(1, np.linalg.norm(V_array, ord='fro') / d))


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def l2_projection(Ui, V, epsilon):
    UV_norm = torch.linalg.vector_norm(torch.matmul(Ui, V), dim=1)
    epsilon_tensor = torch.full((Ui.shape[0],), epsilon).cuda()
    ones = torch.ones_like(epsilon_tensor).cuda()
    factor = torch.max(ones, UV_norm / epsilon_tensor).unsqueeze(1)
    return torch.div(Ui, factor)


def evaluate_low_rank(model, V, U, data):
    test_loss = 0
    test_acc = 0
    n = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(data)):
            X, y = X.cuda(), y.cuda()
            output = model(X + torch.matmul(U[i], V).reshape(X.shape))
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


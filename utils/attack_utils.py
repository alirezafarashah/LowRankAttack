import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class AttackUtils(object):

    def __init__(self, lower_limit, upper_limit, std):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.std = std

    def clamp_operator_norm(self, V):
        V_array = V.detach().cpu().numpy()
        return torch.div(V, max(1, np.linalg.norm(V_array, 2)))

    def evaluate_low_rank(self, model, V, U, test_loader):
        test_loss = 0
        test_acc = 0
        n = 0
        with torch.no_grad():
            for i, (X, y, batch_idx) in enumerate(tqdm(test_loader)):
                X, y = X.cuda(), y.cuda()
                output = model(X + torch.matmul(U[i], V).reshape(X.shape))
                loss = F.cross_entropy(output, y)
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
        return test_loss / n, test_acc / n

    def evaluate_model(self, model, test_loader):
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

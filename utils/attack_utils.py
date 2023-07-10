import torch
import torch.nn.functional as F
from tqdm import tqdm


class AttackUtils(object):

    def __init__(self, lower_limit, upper_limit, std):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.std = std

    def evaluate_low_rank(self, model, V, U, test_loader):
        test_loss = 0
        test_acc = 0
        n = 0
        with torch.no_grad():
            for i, (X, y, batch_idx) in enumerate(tqdm(test_loader)):
                X, y = X.cuda(), y.cuda()
                output = model(X + torch.matmul(U[i].reshape(X.shape[0], -1), V).reshape(X.shape))
                loss = F.cross_entropy(output, y)
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
        return test_loss / n, test_acc / n

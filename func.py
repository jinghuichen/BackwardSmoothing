import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def cw_loss(logits, y, confidence=0):
    onehot_y = torch.nn.functional.one_hot(y, num_classes=logits.shape[1]).float()
    self_loss = F.nll_loss(-logits, y, reduction='none')
    other_loss = torch.max((1 - onehot_y) * logits, dim=1)[0]
    return -torch.mean(torch.clamp(self_loss - other_loss + confidence, 0))


class PGDAttack:
    def __init__(self, epsilon=0.031, num_steps=100, step_size=0.0078, image_constraints=(0, 1)):
        self.boxmin = image_constraints[0]
        self.boxmax = image_constraints[1]
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def grad_proj(self, data, order):
        if order == 'inf':
            return data.sign()
        elif order == '2':
            norm = torch.norm(data.view(len(data), -1), 2, 1, keepdim=True)
            return data / (norm + 1e-8).unsqueeze(2).unsqueeze(3).expand_as(data)

    def eta_proj(self, eta, order, epsilon):
        if order == 'inf':
            return torch.clamp(eta, -epsilon, epsilon)
        elif order == '2':
            norm_eta = torch.norm(eta.view(len(eta), -1), p=2, dim=1, keepdim=True)
            norm_eta = torch.clamp(norm_eta, epsilon, np.inf).unsqueeze(2).unsqueeze(3).expand_as(eta)
            return eta * epsilon / norm_eta

    def attack(self, model, X, y, loss_type='ce'):
        out = model(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        X_pgd = X.clone()
        for i in range(self.num_steps):
            X_pgd = Variable(X_pgd, requires_grad=True)
            if loss_type == 'ce':
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            else:
                loss = cw_loss(model(X_pgd), y)
            loss.backward()
            X_pgd = X_pgd + self.step_size * self.grad_proj(X_pgd.grad.data, 'inf')
            eta = self.eta_proj(X_pgd - X, 'inf', self.epsilon)
            X_pgd = X + eta
            X_pgd = torch.clamp(X_pgd, self.boxmin, self.boxmax)
        err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
        return err.item(), err_pgd.item()



class DataAugmentModel(nn.Module):
    def __init__(self, model, im_mean=None, im_std=None):
        super(DataAugmentModel, self).__init__()
        self.model = model
        self.im_mean = im_mean
        self.im_std = im_std

    def forward(self, image):
        image = (image - self.im_mean.expand_as(image)) / self.im_std.expand_as(image)
        return self.model(image)
          
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def training_loss(model,
                  x_natural,
                  y,
                  optimizer,
                  step_size=0.031,
                  epsilon=0.031,
                  perturb_steps=1,
                  gamma=1.):

    # generate adversarial example
    model.eval()
    x_adv = x_natural.detach() + torch.FloatTensor(
        *x_natural.shape).uniform_(-epsilon, epsilon).cuda()

    for i in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits = model(x_adv)
            loss_inner = F.cross_entropy(logits, y)
            
        grad = torch.autograd.grad(loss_inner, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                                    epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    loss_inner = F.cross_entropy(model(x_adv), y).detach()

    # train model parameters
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    return loss

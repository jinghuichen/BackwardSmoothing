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
                  beta=6.0,
                  gamma=1.
                  ):

    criterion_kl = nn.KLDivLoss(size_average=False)
    batch_size = len(x_natural)

    # generate adversarial example
    model.eval()
    x_adv = x_natural.detach()
    #  + torch.FloatTensor(*x_natural.shape).uniform_(-epsilon, epsilon).cuda()

    x_adv.requires_grad_()
    logits_nat = model(x_natural).detach()
    rand_vector = gamma * torch.FloatTensor(*logits_nat.shape).uniform_(-1., 1.).cuda()
    loss_os = -(1.0 / len(x_natural)) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                     F.softmax(logits_nat + rand_vector, dim=1))
    grad = torch.autograd.grad(loss_os, [x_adv])[0]
    x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
    x_adv = torch.min(torch.max(x_adv, x_natural -
                                epsilon), x_natural + epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_inner = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                           F.softmax(logits_nat, dim=1))
        grad = torch.autograd.grad(loss_inner, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                                    epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # train model parameters
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    logits_nat = model(x_natural)
    loss_natural = F.cross_entropy(logits_nat, y)
    loss_robust = beta * (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                           F.softmax(logits_nat, dim=1))
    loss = loss_natural + loss_robust
    return loss

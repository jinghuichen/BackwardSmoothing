import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta

def l2_norm_batch(v):
    norms = (v ** 2).sum([1, 2, 3]) ** 0.5
    return norms

def get_input_grad(model, X, y, eps, delta_init='none', backprop=False):
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == 'random_corner':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else:
        raise ValueError('wrong delta init')

    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    return grad


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


    reg = torch.zeros(1).cuda()[0]  # for .item() to run correctly
    grad = grad = get_input_grad(model, x_natural, y, epsilon, delta_init='none', backprop=True)
    grad2 = get_input_grad(model, x_natural, y, epsilon, delta_init='random_uniform', backprop=True)
    grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
    grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
    grad1_norms, grad2_norms = l2_norm_batch(grad1), l2_norm_batch(grad2)
    grad1_normalized = grad1 / grad1_norms[:, None, None, None]
    grad2_normalized = grad2 / grad2_norms[:, None, None, None]
    cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
    reg += 0.3 * (1.0 - cos.mean())
    loss += reg

    return loss

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from func import PGDAttack, DataAugmentModel
from utils import progress_bar

from trades_backward import training_loss

import logging
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=1, type=int,
                    help='perturb number of steps')
parser.add_argument('--alpha', default=0.031, type=float,
                    help='perturb step size')
parser.add_argument('--beta', type=float, default=6.0,
                    help='regularization parameter')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='regularization parameter')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-name', default='trades_backward',
                    help='directory of model for saving checkpoint')
parser.add_argument('--arch', default='resnet',
                    help='model architecture')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tiny'],
                    help='dataset')
parser.add_argument('--lr-sch', default='stage',
                    choices=['cyclic', 'stage', 'stage75', 'cosine'])

args = parser.parse_args()
print(args)
out_dir_name = args.model_name + '_steps' + str(args.num_steps) + '_' + args.dataset + '_' + args.lr_sch + str(args.epochs) + '_lr' + str(
    args.lr) + '_alpha' + str(args.alpha) + '_beta' + str(args.beta) + '_gamma' + str(args.gamma) + '_' + args.arch

# settings
if not os.path.exists(out_dir_name):
    os.makedirs(out_dir_name)
logfile = os.path.join(out_dir_name, 'output.log')
if os.path.exists(logfile):
    os.remove(logfile)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(out_dir_name, 'output.log'))
logger.info(args)
logger.info('Epoch \t LR (Nat Acc) \t \t Loss (Rob Acc)')

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

if args.dataset == 'cifar10':
    from data_loader import load_cifar10
    train_loader, test_loader, im_mean, im_std = load_cifar10(
        batch_size=args.batch_size, augment=True)
    num_classes = 10
elif args.dataset == 'cifar100':
    from data_loader import load_cifar100
    train_loader, test_loader, im_mean, im_std = load_cifar100(
        batch_size=args.batch_size, augment=True)
    num_classes = 100
elif args.dataset == 'tiny':
    from data_loader import load_tiny
    train_loader, test_loader, im_mean, im_std = load_tiny(
        batch_size=args.batch_size, augment=True)
    num_classes = 200

if args.arch == 'resnet':
    from models.resnet import ResNet18
    model = ResNet18(num_classes=num_classes).to(device)
elif args.arch == 'wrn':
    from models.wideresnet import WideResNet
    model = WideResNet(num_classes=num_classes).to(device)
elif args.arch == 'densenet':
    from models.densenet import DenseNet121
    model = DenseNet121(num_classes=num_classes).to(device)
elif args.arch == 'resnet_p':
    import torchvision.models as models
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
elif args.arch == 'resnet64':
    from models.resnet64 import ResNet50
    model = ResNet50(num_classes=num_classes).to(device)
model = DataAugmentModel(model, im_mean=im_mean, im_std=im_std)
# model = nn.DataParallel(model)

optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

lr_steps = args.epochs * len(train_loader)
if args.lr_sch == 'cyclic':
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=args.lr * 2,
                                                  step_size_up=1,
                                                  step_size_down=lr_steps)
elif args.lr_sch == 'stage':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
elif args.lr_sch == 'stage75':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[lr_steps * 3 / 4, lr_steps * 9 / 10], gamma=0.1)
elif args.lr_sch == 'cosine':
    def cos_func(steps): return 0.5 * \
        (1 + np.cos((steps - 1) / lr_steps * np.pi))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[cos_func])


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = training_loss(model=model,
                             x_natural=data,
                             y=target,
                             optimizer=optimizer,
                             step_size=args.alpha,
                             epsilon=args.epsilon,
                             perturb_steps=args.num_steps,
                             beta=args.beta,
                             gamma=args.gamma)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_total += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Avg Loss: %.3f Lr: %.3f | Epoch: %d'
                     % (loss.item(), loss_total / (batch_idx + 1), scheduler.get_last_lr()[0], epoch))

    return loss_total / (batch_idx + 1)


def eval_adv(model, device, test_loader, steps=20, step_size=0.0078, loss_type='ce', batch_num=None):
    model.eval()
    attacker = PGDAttack(epsilon=args.epsilon,
                         num_steps=steps, step_size=step_size)

    robust_err_total = 0
    natural_err_total = 0
    total = 0

    total_batch = len(test_loader)
    if batch_num is not None:
        total_batch = batch_num

    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_num is not None and batch_idx >= batch_num:
            break
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = attacker.attack(model, X, y, loss_type)
        robust_err_total += err_robust
        natural_err_total += err_natural
        total += len(data)

        progress_bar(batch_idx, total_batch, 'Nat Acc: %.4f | Rob Acc: %.4f'
                     % (1 - natural_err_total / total, 1 - robust_err_total / total))

    nat_acc = 1 - natural_err_total / total
    robust_acc = 1 - robust_err_total / total

    return nat_acc, robust_acc


def main():
    total_time = 0
    best = 0
    best_epoch = None
    for epoch in range(1, args.epochs + 1):

        # adversarial training
        start = time.time()
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        logger.info('Train: %d \t %.4f \t  \t %.4f',
                    epoch, scheduler.get_last_lr()[0], train_loss)
        total_time += time.time() - start

        # evaluation
        nat_acc, robust_acc = eval_adv(model, device, test_loader, steps=20,
                                       step_size=0.0078, loss_type='ce', batch_num=20)
        logger.info('CE   : %d \t %.4f \t  \t %.4f ', epoch, nat_acc, robust_acc)
        nat_acc, robust_cw = eval_adv(
            model, device, test_loader, steps=20, step_size=0.0078, loss_type='cw', batch_num=20)
        logger.info('CW   : %d \t %.4f \t  \t %.4f ', epoch, nat_acc, robust_cw)

        # update best
        if robust_acc > best:
            best = robust_acc
            best_epoch = epoch
            print('current best!')
            # save checkpoint
            torch.save(model.state_dict(),
                       os.path.join(out_dir_name, 'model-best.pt'))

    print('Total time used:', total_time,
          'seconds', total_time / 60, 'minutes')
    logger.info('Total training time: %.4f secends, %.4f minutes',
                total_time, total_time/60)

    # final evaluation
    print('pgd100 evaluation on best epoch', best_epoch)
    model.load_state_dict(torch.load(os.path.join(
        out_dir_name, 'model-best.pt')))
    nat_acc, robust_pgd = eval_adv(
        model, device, test_loader, steps=100, step_size=0.0078, loss_type='ce')
    nat_acc, robust_cw = eval_adv(
        model, device, test_loader, steps=100, step_size=0.0078, loss_type='cw')
    logger.info('CE-100: %d \t %.4f \t  \t %.4f ',
                best_epoch, nat_acc, robust_pgd)
    logger.info('CW-100: %d \t %.4f \t  \t %.4f ',
                best_epoch, nat_acc, robust_cw)


if __name__ == '__main__':
    main()

# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP) and ABC(https://github.com/LeeHyuck/ABC)
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math
import numpy as np
import wideresnetwithABC as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int, help='batchsize of labeled training data')
parser.add_argument('--unlabeledbatch', default=2, type=float,help='batchsize ratio of unlabeled data to labeled data')
parser.add_argument('--lr', '--learning-rate', default=0.0015, type=float, help='initial learning rate')
parser.add_argument('--wd', '--weight-decay', default=0.04, type=float, help='weight decay')
# Checkpoints
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result', help='Directory to output the result')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--num_max', type=int, default=1500, help='Number of samples in the maximal class of labeled data')
parser.add_argument('--num_max_u', type=int, default=3000, help='Number of samples in the maximal class of unlabeled data')
parser.add_argument('--imb_ratio', type=int, default=100, help='Imbalance ratio of labeled data')
parser.add_argument('--imb_ratio_u', type=int, default=100, help='Imbalance ratio of unlabeled data')
parser.add_argument('--val-iteration', type=int, default=500, help='Frequency for the evaluation')
# Hyperparameters for FixMatch
parser.add_argument('--ema-decay', default=0.999, type=float)
# dataset and imbalanced type
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
parser.add_argument('--imbalancetype', type=str, default='long', help='Long tailed or step imbalanced')
#RECD
parser.add_argument('--AAFM', default=1.5, type=float, help='scale hyperparameter for AAFM')
parser.add_argument('--power', default=2, type=float, help='power of confidence')
parser.add_argument('--warm', default=20, type=int, help='warm up epoch')



args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
if args.dataset == 'cifar10':
    import dataset.fix_cifar10 as dataset

    print(f'==> Preparing imbalanced CIFAR10')
    num_class = 10

elif args.dataset == 'stl10':
    import dataset.fix_stl10 as dataset

    print(f'==> Preparing imbalanced STL-10')
    num_class = 10

elif args.dataset == 'cifar100':
    import dataset.fix_cifar100 as dataset
    print(f'==> Preparing imbalanced CIFAR100')
    num_class = 100
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()



if args.manualSeed == 100:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(torch.__version__)


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio, args.imbalancetype)
    U_SAMPLES_PER_CLASS = make_imb_data(args.num_max_u, num_class, args.imb_ratio_u, args.imbalancetype)
    if args.dataset == 'cifar10':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar10('./data', N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS)
    elif args.dataset == 'stl10':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_stl10('./data', N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS)
    elif args.dataset == 'cifar100':
        train_labeled_set, train_unlabeled_set,  test_set = dataset.get_cifar100('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)
    else :
        train_labeled_set , train_unlabeled_set, test_set = False

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=int(args.unlabeledbatch * args.batch_size), shuffle=True, num_workers=4, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if np.array(N_SAMPLES_PER_CLASS).sum() + np.array(U_SAMPLES_PER_CLASS).sum() >= 30000 or args.dataset == 'stl10' :
        args.wd = 0.01
        args.AAFM = 0.5
    elif args.dataset == 'cifar100':
        args.wd = 0.06

    # Model
    print("==> creating WRN-28-2 with abc")

    def create_model(ema=False):
        model = models.WideResNet(num_classes=num_class)
        model = model.cuda()

        params = list(model.parameters())
        if ema:
            for param in params:
                param.detach_()

        return model, params

    model, params = create_model()
    ema_model, _ = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in params) / 1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 1

    # Resume
    title = 'CDRA_fix-' + args.dataset
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']+1
        weightu = checkpoint['weightu']
        confidence = checkpoint['confidence']
        confidenceu = checkpoint['confidenceu']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['epoch', 'Test Acc.','Train Loss', 'Train Loss X', 'Train Loss U', 'abcloss', 'Test Loss'])
        weightu = torch.tensor(np.array(N_SAMPLES_PER_CLASS)/sum(N_SAMPLES_PER_CLASS)).cuda()
        confidence = torch.zeros(num_class).cuda()
        confidenceu = torch.zeros(num_class).cuda()


    weightu=weightu.type(torch.cuda.FloatTensor)
    warm = args.warm

    meanacc = 0
    meangm = 0
    for epoch in range(start_epoch, args.epochs+1):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, state['lr']))

        # Training part
        train_loss, train_loss_x, train_loss_u, abcloss, weightu, confidence, confidenceu, \
            = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion,
            weightu,torch.tensor(N_SAMPLES_PER_CLASS).cuda(),epoch, confidence, confidenceu, warm)

        test_loss, test_acc, testclassacc, GM = validate(test_loader, ema_model, criterion, mode='Test Stats ')

        if args.dataset == 'cifar10':
            print("each class accuracy test", testclassacc, ", GM",np.round(GM.item(),4)*100)

        elif args.dataset == 'stl10':
            print("each class accuracy test", testclassacc, ", GM",np.round(GM.item(),4)*100)

        elif args.dataset == 'cifar100':
            print("each class accuracy test", testclassacc, ", GM",np.round(GM.item(),4)*100)

        logger.append([epoch, test_acc,train_loss, train_loss_x, train_loss_u, abcloss, test_loss])
        # Save models
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'weightu' : weightu,
            'optimizer': optimizer.state_dict(),
            'confidence' : confidence,
            'confidenceu' : confidenceu,
        }, epoch)

        if epoch >= 481:
            meanacc = meanacc + 0.05*test_acc
            meangm = meangm + 0.05*GM
        if epoch == args.epochs:
            print('meanacc :', np.round(meanacc,4), 'meangm :', np.round(meangm.item()*100,4))
    logger.close()


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, weightu, N_SAMPLES_PER_CLASS,epoch, confidence, confidenceu, warm):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_abc = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()

    new_confidence = confidence
    new_confidenceu = confidenceu

    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = unlabeled_train_iter.next()


        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x2 = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1, 1), 1)

        inputs_x, targets_x2 = inputs_x.cuda(), targets_x2.cuda(non_blocking=True)
        inputs_u, inputs_u2, inputs_u3 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()


        # Generate the pseudo labels
        with torch.no_grad():
            q1 = model(inputs_u)
            outputs_u_abc = model.classify2(q1)
            targets_u2_abc = torch.softmax(outputs_u_abc, dim=1).detach()
        max_p_abc, p_hat_abc = torch.max(targets_u2_abc, dim=1)
        p_hat_abc = torch.zeros(inputs_u2.size(0), num_class).cuda().scatter_(1, p_hat_abc.view(-1, 1), 1)

        # RECD
        weightu = 0.99 * weightu + 0.01 * targets_u2_abc.mean(0)

        q = model(inputs_x)
        q2 = model(inputs_u2)
        q3 = model(inputs_u3)

        # obtaining confidence
        with torch.no_grad():
            logitsx = F.softmax(model.classify2(q))
            added_new_confidence = torch.sum(logitsx * targets_x2, dim=0) / torch.sum(targets_x2, dim=0)
            added_new_confidenceu = torch.sum(targets_u2_abc * p_hat_abc, dim=0) / torch.sum(p_hat_abc, dim=0)

            non_nan = torch.logical_not(torch.isnan(added_new_confidence))
            new_confidence[non_nan == True] = 0.9 * new_confidence[non_nan == True] + 0.1 * added_new_confidence.cuda()[non_nan == True]
            non_nan = torch.logical_not(torch.isnan(added_new_confidenceu))
            new_confidenceu[non_nan == True] = 0.9 * new_confidenceu[non_nan == True] + 0.1 * added_new_confidenceu.cuda()[non_nan == True]

        # AAFM
        AAFM_l = N_SAMPLES_PER_CLASS / torch.max(N_SAMPLES_PER_CLASS)
        AAFM_l = AAFM_l*new_confidence**args.power
        AAFM_l = torch.sum(targets_x2.cuda() * torch.tensor(AAFM_l).cuda(), dim=1).detach()

        AAFM_u = weightu / torch.max(weightu)
        AAFM_u = AAFM_u * new_confidenceu**args.power
        AAFM_u = torch.sum(targets_u2_abc.cuda() * torch.tensor(AAFM_u).cuda(), dim=1).detach()

        # applying AAFM
        if epoch > warm:
            q = q * (1 + args.AAFM * AAFM_l.type(torch.cuda.FloatTensor)).view(inputs_x.size(0),1).repeat(1, 128)
            q2 = q2 * (1 + args.AAFM * AAFM_u).view(inputs_u2.size(0), 1).repeat(1, 128)
            q3 = q3 * (1 + args.AAFM * AAFM_u).view(inputs_u2.size(0), 1).repeat(1, 128)


        # FixMatch
        all_targets = torch.cat([targets_x2, p_hat_abc, p_hat_abc], dim=0)

        logits_x_fix = model.classify(q)
        logits_u2_fix = model.classify(q2)
        logits_u3_fix = model.classify(q3)
        logits_u_fix = torch.cat([logits_u2_fix, logits_u3_fix], dim=0)

        Lx, Lu = criterion(logits_x_fix, all_targets[:batch_size], logits_u_fix, all_targets[batch_size:])


        # ABC
        logits_x_abc = model.classify2(q)
        logits_u2_abc = model.classify2(q2)
        logits_u3_abc = model.classify2(q3)

        max_p2, label_u = torch.max(targets_u2_abc, dim=1)
        label_u = torch.zeros(inputs_u2.size(0), num_class).scatter_(1, label_u.cpu().view(-1, 1),1)
        label_u = label_u.cuda()

        weightl1 = torch.min(N_SAMPLES_PER_CLASS) / N_SAMPLES_PER_CLASS
        weightl1 = torch.sum(targets_x2.cuda() * torch.tensor(weightl1).cuda(), dim=1)
        maskforbalance = torch.bernoulli(weightl1)

        weightu1 = torch.min(weightu) / (weightu)
        weightu1 = torch.sum(label_u.cuda() * torch.tensor(weightu1).cuda(0), dim=1)
        maskforbalanceu1 = torch.bernoulli(weightu1)
        maskforbalanceu2 = torch.bernoulli(weightu1)

        abcloss = -torch.mean(maskforbalance.detach() * torch.sum(torch.log(F.softmax(logits_x_abc) + 1e-8) * targets_x2.cuda(), dim=1))
        abcloss1 = -torch.mean(maskforbalanceu1.detach() * torch.sum(torch.log(F.softmax(logits_u2_abc) + 1e-8) * targets_u2_abc.detach(),dim=1))
        abcloss2 = -torch.mean(maskforbalanceu2.detach() * torch.sum(torch.log(F.softmax(logits_u3_abc) + 1e-8) * targets_u2_abc.detach(),dim=1))

        totalabcloss = abcloss + abcloss1 + abcloss2

        loss = Lx + Lu + totalabcloss

        # record loss
        losses.update(loss.item(), inputs_x.size(0)+inputs_u2.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_u2.size(0))
        losses_abc.update(abcloss.item(), inputs_x.size(0)+inputs_u2.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                     'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}| Loss_m: {loss_m:.4f}'.format(
            batch=batch_idx + 1,
            size=args.val_iteration,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            loss_m=losses_abc.avg,
        )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg, losses_abc.avg,weightu, new_confidence, new_confidenceu)


def validate(valloader, model, criterion, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    accperclass = np.zeros((num_class))
    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    section_acc = torch.zeros(3)
    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):

            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            targetsonehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, targets.cpu().view(-1, 1).long(), 1)
            q = model(inputs)
            outputs2 = model.classify2(q)
            unbiasedscore = F.softmax(outputs2)
            unbiased = torch.argmax(unbiasedscore, dim=1)
            outputs2onehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, unbiased.cpu().view(-1, 1).long(), 1)
            loss = criterion(outputs2, targets)
            accperclass = accperclass + torch.sum(targetsonehot * outputs2onehot, dim=0).cpu().detach().numpy().astype(
                np.int64)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs2, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # classwise prediction
            pred_label = outputs2.max(1)[1]
            pred_mask = (targets == pred_label).float().cuda()
            classwise_correct = classwise_correct.cuda()
            classwise_num = classwise_num.cuda()
            for i in range(num_class):
                class_mask = (targets == i).float().cuda()

                classwise_correct[i] += (class_mask * pred_mask).sum()
                classwise_num[i] += class_mask.sum()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                         'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(valloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()

    section_num = int(num_class / 3)
    classwise_acc = (classwise_correct / classwise_num)
    section_acc[0] = classwise_acc[:section_num].mean()
    section_acc[2] = classwise_acc[-1 * section_num:].mean()
    section_acc[1] = classwise_acc[section_num:-1 * section_num].mean()
    GM = 1
    for i in range(num_class):
        if classwise_acc[i] == 0:
            # To prevent the N/A values, we set the minimum value as 0.001
            GM *= (1 / (100 * num_class)) ** (1 / num_class)
        else:
            GM *= (classwise_acc[i]) ** (1 / num_class)

    if args.dataset == 'cifar10':
        accperclass = accperclass / 1000
    elif args.dataset == 'stl10':
        accperclass = accperclass / 800
    elif args.dataset == 'cifar100':
        accperclass = accperclass / 100

    return (losses.avg, top1.avg, accperclass, GM)



def make_imb_data(max_num, class_num, gamma, imb):
    if imb == 'long':
        mu = np.power(1 / gamma, 1 / (class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
    if imb == 'step':
        class_num_list = []
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
    return list(class_num_list)


def save_checkpoint(state, epoch, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if epoch == 200 or epoch == 500:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_' + str(epoch) + '.pth.tar'))



class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u):
        Lx = -torch.mean(torch.sum(torch.log(F.softmax(outputs_x, dim=1) + 1e-8) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(torch.log(F.softmax(outputs_u, dim=1) + 1e-8) * targets_u, dim=1))
        return Lx, Lu


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = args.wd * args.lr
        #self.wd = args.wd * 0.002

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)



if __name__ == '__main__':
    main()
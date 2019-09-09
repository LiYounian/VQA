import sys
import os.path
import argparse
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data

if config.model_type == 'baseline':
    import baseline_model as model
elif config.model_type == 'ban':
    import ban_model as model
elif config.model_type == 'finalmodel':
    import final_model as model
import utils


def run(net, loader, optimizer, scheduler, tracker, train=False, has_answers=True, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    assert not (train and not has_answers)
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    # set learning rate decay policy
    if epoch < len(config.gradual_warmup_steps) and config.schedule_method == 'warm_up':
        utils.set_lr(optimizer, config.gradual_warmup_steps[epoch])
        utils.print_lr(optimizer, prefix, epoch)
    elif (epoch in config.lr_decay_epochs) and train and config.schedule_method == 'warm_up':
        utils.decay_lr(optimizer, config.lr_decay_rate)
        utils.print_lr(optimizer, prefix, epoch)
    else:
        utils.print_lr(optimizer, prefix, epoch)

    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    for v, q, a, b, idx, v_mask, q_mask, q_len in loader:
        var_params = {
            'requires_grad': False,
        }
        v = Variable(v.cuda(), **var_params)
        q = Variable(q.cuda(), **var_params)
        a = Variable(a.cuda(), **var_params)
        b = Variable(b.cuda(), **var_params)
        q_len = Variable(q_len.cuda(), **var_params)
        v_mask = Variable(v_mask.cuda(), **var_params)
        q_mask = Variable(q_mask.cuda(), **var_params)

        out = net(v, b, q, v_mask, q_mask, q_len)
        if has_answers:
            answer = utils.process_answer(a)
            loss = utils.calculate_loss(answer, out, method=config.loss_method)
            acc = utils.batch_accuracy(out, answer).data.cpu()

        if train:
            optimizer.zero_grad()
            loss.backward()
            # print gradient
            if config.print_gradient:
                utils.print_grad([(n, p) for n, p in net.named_parameters() if p.grad is not None])
            # clip gradient
            clip_grad_norm_(net.parameters(), config.clip_value)
            optimizer.step()
            if (config.schedule_method == 'batch_decay'):
                scheduler.step()
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            if has_answers:
                accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        if has_answers:
            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    if not train:
        answ = list(torch.cat(answ, dim=0))
        if has_answers:
            accs = list(torch.cat(accs, dim=0))
        else:
            accs = []
        idxs = list(torch.cat(idxs, dim=0))
        # print('{} E{:03d}:'.format(prefix, epoch), ' Total num: ', len(accs))
        # print('{} E{:03d}:'.format(prefix, epoch), ' Average Score: ', float(sum(accs) / len(accs)))
        return answ, accs, idxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='*')
    parser.add_argument('--eval', dest='eval_only', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trainval', action='store_true')
    parser.add_argument('--resume', nargs='*')
    parser.add_argument('--describe', type=str, default='describe your setting')
    args = parser.parse_args()

    print('-' * 50)
    print(args)
    config.print_param()

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    train_loader = data.get_loader(trainval=True)
    print(train_loader.dataset)


if __name__ == '__main__':
    main()


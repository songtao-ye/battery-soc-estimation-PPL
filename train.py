# trian
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from data_loader import DataTrain, DataTest
from data_manager import LgData
from models.model import PMTN as net

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, default='dataset')
argparser.add_argument('--save_path', type=str, default='log')
argparser.add_argument('--SOCOCV', type=bool, default=True)

argparser.add_argument('--seq_len', type=int, default=300)
argparser.add_argument('--interval', type=int, default=50)

argparser.add_argument('--patch_len', type=int, default=10)
argparser.add_argument('--feat_dim', type=int, default=64)

argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--lr', type=float, default=1e-3)
argparser.add_argument('--max_epoch', type=int, default=100)
argparser.add_argument('--num_workers', type=int, default=0)
argparser.add_argument('--shuffle', type=bool, default=True)

args = argparser.parse_args()

def train(model, train_loader, criterion_SOC, optimizer, epoch, use_gpu=True):
    model.train()
    losses = []
    for i, (data, SOC, SOCocv) in enumerate(train_loader):
        if use_gpu:
            data = data.float().cuda()
            SOC = SOC.float().cuda().squeeze()
            SOCocv = SOCocv.float().cuda()
        SOC = SOC[:,(args.patch_len-1)::args.patch_len]
        # SOC = SOC[:, -1].view(-1, 1)
        out_SOC = model(data, SOCocv)

        loss = criterion_SOC(out_SOC, SOC)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print('Epoch: {},  Loss: {:.6f} ({:.6f})'.
          format(epoch, loss.item(), np.mean(losses)))

    return np.mean(losses)


# test函数
def test(model, test_loader, criterion_SOC, epoch, use_gpu=True):
    model.eval()
    iter_preds = []
    iter_labels = []
    losses = []
    preds = defaultdict(list)
    labels = defaultdict(list)
    keys = set()
    record_index = []
    keys_list = []
    with torch.no_grad():
        for i, (data, SOC, key, SOCocv) in enumerate(test_loader):
            if use_gpu:
                data = data.float().cuda()
                SOC = SOC.float().cuda().squeeze()
                SOCocv = SOCocv.float().cuda()

            SOC = SOC[:, (args.patch_len - 1)::args.patch_len]
            # SOC = SOC[:, -1].view(-1, 1)
            out_SOC = model(data, SOCocv)
            loss = criterion_SOC(out_SOC, SOC)

            out_SOC[out_SOC>1] = 1
            out_SOC[out_SOC<0] = 0

            iter_preds.append(out_SOC[:, -1].view(-1, 1))
            iter_labels.append(SOC[:, -1].view(-1, 1))
            losses.append(loss.item())

            # key = key.tolist()
            if keys ^ set(key):  # 如果集合中出现了不同元素，再去寻找是哪个
                for j, k in enumerate(key):
                    if k not in keys:
                        keys.add(k)
                        keys_list.append(k)
                        record_index.append(i * args.batch_size + j)

    record_index.append(len(test_loader.dataset))
    iter_preds = torch.cat(iter_preds, dim=0).cpu().numpy()
    iter_labels = torch.cat(iter_labels, dim=0).cpu().numpy()
    result = evaluation(iter_labels, iter_preds)

    for i, k in enumerate(keys_list):
        preds[k] = iter_preds[record_index[i]:record_index[i + 1]]
        labels[k] = iter_labels[record_index[i]:record_index[i + 1]]

    return np.mean(losses), preds, labels, result


def main():
    pin_memory = False
    torch.random.manual_seed(1234)
    args.save_path = osp.join('log',
                              str(args.seq_len)+ '_'
                              + str(args.interval) + '_'
                              + str(args.patch_len))
    if args.SOCOCV:
        args.save_path = args.save_path + '_SOCOCV'

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # sys.stdout = Logger(osp.join(args.save_path, 'log_train.txt'))

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('Use GPU!')
        pin_memory = True

    lg_data = LgData(base_path = args.data_path, steps = args.seq_len, interval = args.interval)

    train_dataset = DataTrain([lg_data.train_x, lg_data.train_y, lg_data.train_SOCOCVs])
    test_dataset = DataTest([lg_data.test_x, lg_data.test_y, lg_data.test_k, lg_data.test_SOCOCVs])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=pin_memory,
                                               shuffle=args.shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=pin_memory,
                                              drop_last=True,
                                              shuffle=False)

    model = net(d_model=args.feat_dim, patch_len=args.patch_len, context_length=args.seq_len, SOCOCV = args.SOCOCV)

    if use_gpu:
        model = model.cuda()

    #计算参数量，单位M
    params = 0
    for name, parameter in model.named_parameters():
        if 'integration' in name:
            params += parameter.numel()
    print(params/1000)
    print('Total params: %.4fK' % (sum(p.numel() for p in model.parameters()) / 1000.0))

    criterion_SOC = nn.MSELoss()
    # criterion_SOC = nn.HuberLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    test_losses = []
    min_rmse = 100
    print('Start training...')
    for epoch in range(args.max_epoch):
        train_loss = train(model, train_loader, criterion_SOC, optimizer, epoch, use_gpu=use_gpu)

        adjust_learning_rate(optimizer, epoch + 1, args.lr)

        test_loss, preds, labels, results_all = test(model, test_loader, criterion_SOC, epoch,use_gpu=use_gpu)

        results = defaultdict(list)
        for key in preds.keys():
            results[key] = evaluation(labels[key], preds[key])

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print('-'*25)
        print('Epoch={}, train_loss={:.6f}, test_loss={:.6f}'
              .format(epoch, train_loss, test_loss))
        for key in results.keys():
            print('{}: mae = {:.6f}, mse = {:.6f}, rmse = {:.6f}, r2 = {:.6f}'
                  .format(key, results[key][0], results[key][1], results[key][2], results[key][3]))

        # preds_all = np.concatenate([preds[key] for key in preds.keys()], axis=0)
        # labels_all = np.concatenate([labels[key] for key in labels.keys()], axis=0)
        # results_all = evaluation(labels_all, preds_all)

        print('mean mae = {:.6f}, mean mse = {:.6f}, mean rmse = {:.6f}, mean r2 = {:.6f}'
              .format(results_all[0], results_all[1], results_all[2], results_all[3]))
        print('-'*25)

        if results_all[2] < min_rmse:
            min_rmse = results_all[2]
            plot_results(preds, labels, results, save_path=osp.join(args.save_path, 'result.png'))
            torch.save(model.state_dict(), osp.join(args.save_path, 'best_model.pth.tar'))

    plot_loss(train_losses, test_losses, save_path=args.save_path)

if __name__ == '__main__':
    main()

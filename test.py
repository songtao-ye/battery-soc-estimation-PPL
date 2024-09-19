# test
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from data_loader import DataTrain, DataTest
from data_manager import LgData
from models.model import PMTN as net

import time

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, default='dataset')
argparser.add_argument('--save_path', type=str, default='log')
argparser.add_argument('--SOCOCV', type=bool, default=True)
# 300_50_10_SOCOCV
argparser.add_argument('--seq_len', type=int, default=300)
argparser.add_argument('--interval', type=int, default=50)

argparser.add_argument('--patch_len', type=int, default=10)
argparser.add_argument('--feat_dim', type=int, default=64)

argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--num_workers', type=int, default=0)

args = argparser.parse_args()

# test函数
def test(model, test_loader, use_gpu=True):
    model.eval()
    iter_preds = []
    iter_labels = []
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
            else:
                data = data.float()
                SOC = SOC.float().squeeze()
                SOCocv = SOCocv.float()

            SOC = SOC[:, (args.patch_len - 1)::args.patch_len]
            # SOC = SOC[:, -1].view(-1, 1)
            strat = time.time()
            out_SOC = model(data, SOCocv)
            # print('time:', (time.time()-strat)*1000, 'ms')

            out_SOC[out_SOC>1] = 1
            out_SOC[out_SOC<0] = 0

            iter_preds.append(out_SOC[:, -1].view(-1, 1))
            iter_labels.append(SOC[:, -1].view(-1, 1))

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

    return preds, labels, result


def main():
    pin_memory = False
    torch.random.manual_seed(1234)

    args.save_path = osp.join(args.save_path,
                              str(args.seq_len)+ '_'
                              + str(args.interval) + '_'
                              + str(args.patch_len))
    if args.SOCOCV:
        args.save_path = args.save_path + '_SOCOCV'
    args.model_path = osp.join(args.save_path, 'best_model.pth.tar')

    use_gpu = torch.cuda.is_available()
    use_gpu = False
    if use_gpu:
        print('Use GPU!')
        pin_memory = True

    lg_data = LgData(base_path = args.data_path, steps = args.seq_len, interval = args.seq_len)

    test_dataset = DataTest([lg_data.test_x, lg_data.test_y, lg_data.test_k, lg_data.test_SOCOCVs])

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=pin_memory,
                                              drop_last=True,
                                              shuffle=False)

    model = net(d_model=args.feat_dim, patch_len=args.patch_len, context_length=args.seq_len, SOCOCV = args.SOCOCV)

    if use_gpu:
        model.load_state_dict(torch.load(args.model_path))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    #计算参数量，单位M
    params = 0
    for name, parameter in model.named_parameters():
        if 'integration' in name:
            params += parameter.numel()
    print(params/1000)
    print('Total params: %.4fK' % (sum(p.numel() for p in model.parameters()) / 1000.0))

    preds, labels, results_all = test(model, test_loader,use_gpu=use_gpu)

    results_i = defaultdict(list)
    for key in preds.keys():
        results_i[key] = evaluation(labels[key], preds[key])

    #保存preds和labels
    np.save(osp.join(args.save_path, 'preds.npy'), preds)
    np.save(osp.join(args.save_path, 'labels.npy'), labels)

    #将同样的温度进行合并
    dict_preds = defaultdict(list)
    dict_labels = defaultdict(list)
    for key, value in preds.items():
        dict_preds[key.split('_')[0]].extend(value)
    for key, value in labels.items():
        dict_labels[key.split('_')[0]].extend(value)

    for key in dict_preds.keys():
        dict_preds[key] = np.array(dict_preds[key])
        dict_labels[key] = np.array(dict_labels[key])

    results = defaultdict(list)
    for key in dict_preds.keys():
        results[key] = evaluation(dict_labels[key], dict_preds[key])

    for key in results.keys():
        print('{}: mae = {:.6f}, mse = {:.6f}, rmse = {:.6f}, r2 = {:.6f}'
              .format(key, results[key][0], results[key][1], results[key][2], results[key][3]))

    print('mean mae = {:.6f}, mean mse = {:.6f}, mean rmse = {:.6f}, mean r2 = {:.6f}'
          .format(results_all[0], results_all[1], results_all[2], results_all[3]))
    print('-'*25)

    plot_results(preds, labels, results_i, save_path=osp.join(args.save_path, 'result.png'))


if __name__ == '__main__':
    main()

# 指标评估 MAE, MSE, RMSE
import os
import sys
import errno
import os.path as osp
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
import random
import numpy as np

from matplotlib import pyplot as plt


def adjust_learning_rate(optimizer, epoch, learning_rate):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch-1) // 20))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def evaluation(true, pred):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = sqrt(mse)
    r2 = r2_score(true, pred)
    return mae, mse, rmse, r2

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def plot_loss(train_losses, test_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'r', label='train')
    plt.plot(test_losses, 'b', label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(osp.join(save_path,'loss.png'))
    # plt.show()

def random_plot_SOC_pred(dict_SOC, num = 4, save_path = 'result.png'):
    data = random.sample(dict_SOC.items(), num)
    plt.figure(figsize=(10, 5))
    for i, (key, value) in enumerate(data):
        plt.subplot(2, 2, i + 1)
        plt.plot(value['labelmAh'], 'r.', label='true')
        plt.plot(value['predmAh'], 'b.', label='predict')
        plt.xlabel('time (s)')
        plt.ylabel('remaining capacity (mAh)')
        plt.title(key)
        plt.legend()
    plt.savefig(save_path)
    # plt.show()

def get_mean_std_gt(dict_SOH):
    # 计算每个循环的预测值的均值和标准差
    means = []
    stds = []
    gt = []
    for key, value in dict_SOH.items():
        pred = value['predmAh']
        mean = np.mean(pred, axis=0)
        std = np.std(pred, axis=0)
        means.append(mean)
        stds.append(std)
        gt.append(value['labelmAh'][0])

    return np.array(means), np.array(stds), np.array(gt)


def plot_SOH_pred_mean_std(dict_SOH, save_path = 'result.png'):
    means, stds, gt = get_mean_std_gt(dict_SOH)
    plt.figure(figsize=(10, 5))
    plt.plot(gt, 'r', label='true')
    plt.plot(means, 'b', label='predict')
    r1 = (means - stds).flatten().tolist()
    r2 = (means + stds).flatten().tolist()
    plt.fill_between(range(len(r1)), r1, r2, facecolor='blue', alpha=0.3)
    plt.xlabel('cycle')
    plt.ylabel('maximum capacity (mAh)')
    plt.legend()
    plt.savefig(save_path)
    # plt.show()



def plot_results(pred, true, results, save_path = None):
    # 可视化预测结果
    # 6副子图
    plt.figure(figsize=(12, 24))
    for i, key in enumerate(pred.keys()):
        rmse = results[key][2]
        plt.subplot(6, 3, i + 1)
        plt.plot(true[key], 'r.', label='true')
        plt.plot(pred[key], 'b.', label='predict')
        plt.xlabel('Step')
        plt.ylabel('SOC')
        plt.title('{} RMSE {:.4f}'.format(key, rmse))
        plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()

    # plt.figure(figsize=(30, 5))
    # plt.plot(true[:,0], 'r.', label='true')
    # plt.plot(pred[:,0], 'b.', label='predict')
    #
    # # plt.plot(range(window_size + pred.shape[0]-mutil_step, window_size + pred.shape[0]), pred[-4], 'y.', label='_predict')
    # plt.xlabel('Step')
    # plt.ylabel('SOC')
    # plt.legend()
    # if save_path is not None:
    #     plt.savefig(save_path)
    # plt.show()


def inverse_transform_col(scaler, y, n_col):
    # 将归一化后的数据转换为原始数据
    y = y.copy()
    y -= scaler.min_[n_col]
    y /= scaler.scale_[n_col]
    return y


from __future__ import print_function, division
import os
import argparse
import torch.nn as nn
import numpy as np

from network.CSNet_Layers import ADMMESINetLayer
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from utils.dataset import EEGDataset
import time
from torch.utils.data import DataLoader
from scipy.io import loadmat
from utils.fftc import *
from os.path import join
from datetime import datetime

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, input, target):
        return torch.abs(input - target).mean()

def init_ddp(local_rank):
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='gloo', init_method='env://')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reduce_tensor(tensor: torch.Tensor):

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt


def get_ddp_generator(seed=3407):

    local_rank = int(os.environ['LOCAL_RANK'])
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g

def prepare():
    parser = argparse.ArgumentParser(description=' main ')
    parser.add_argument('--train_dir', default='your train dataset file path', type=str,
                        help='directory of data')
    parser.add_argument('--val_dir', default='your validate dataset file path', type=str,
                        help='directory of data')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=200, type=int, help='number of epochs')
    parser.add_argument('--outf', type=str, default='logs_net', help='path of log files')
    parser.add_argument('--best_model', type=str, default='best_net', help='path of log files')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--gpu', default='0,1', help='gpu device ids for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args

def main(args):
    dist.init_process_group(backend='gloo')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    dir = 'your lead field matrix file path'
    data = loadmat(join(dir, os.listdir(dir)[0]))
    L_data = data['Gain']
    L = torch.Tensor(L_data)
    lambda1 = 0.1
    lambda2 = 1.5
    delta = 0.01
    model = ADMMESINetLayer(L).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    MSE = torch.nn.MSELoss(reduction='mean').cuda()
    MAE = MAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-6)


    train_data = EEGDataset(args.train_dir)
    val_data = EEGDataset(args.val_dir)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)


    avg_train_loss = []
    avg_validation_loss = []
    len_train = len(train_loader)
    len_val = len(val_loader)
    if local_rank == 0:
        print("len_train: ", len_train, "\tlen_val: ", len_val)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = 1000000
    patience = 2
    counter = 0

    for epoch in range(args.num_epoch):
        if local_rank == 0:
            print('EPOCH {}:'.format(epoch + 1))
        start_time = time.time()
        model.train(True)
        running_loss = 0.0
        last_loss = 0.0
        ###############################################################################################学习率
        # adjust_learning_rate(optimizer, epoch, lr=0.002)
        for i, data in enumerate(train_loader):
            # 获取数据----------------------------------------------------------
            source_data, eeg_data = data
            eeg_data = eeg_data.to('cuda')
            source_data = source_data.to('cuda')

            # 将数据送入网络-----------------------------------------------------
            optimizer.zero_grad()
            gen_source_data = model(eeg_data)
            gen_B_data = torch.matmul(L.to('cuda'), gen_source_data)
            loss_B_mse = lambda1 * MSE(gen_B_data, eeg_data)
            loss_s_mse = lambda2 * MSE(gen_source_data, source_data)
            loss_s_mae = lambda2 * delta * MAE(gen_source_data, source_data)

            loss = loss_s_mae + loss_s_mse + loss_B_mse

            # 计算loss并反向传播-----------------------------------------------------
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            last_loss = running_loss / len(train_loader)


            # validation ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for i, vdata in enumerate(val_loader):
                    source_data, eeg_data = vdata
                    eeg_data = eeg_data.to('cuda')
                    source_data = source_data.to('cuda')
                    gen_source_data = model(eeg_data)
                    gen_B_data = torch.matmul(L, gen_source_data)
                    loss_B_mse = lambda1 * MSE(gen_B_data, eeg_data)
                    loss_s_mse = lambda2 * MSE(gen_source_data, source_data)
                    loss_s_mae = lambda2 * delta * MAE(gen_source_data, source_data)
                    vloss = loss_s_mae + loss_s_mse + loss_B_mse
                    running_val_loss += vloss.item()
            avg_val_loss = running_val_loss / len(val_loader)
            end_time = time.time()
            total_time = end_time - start_time
            if local_rank == 0:
                print(' Time : {}, LOSS train {} valid {}'.format(total_time, last_loss, avg_val_loss))
            avg_train_loss.append(last_loss)
            avg_validation_loss.append(avg_val_loss)

            if avg_val_loss < best_vloss:
                best_vloss = avg_val_loss
                model_path = 'model_{}_{}.pth'.format(timestamp, epoch + 1)
                best_model_state = model.state_dict()
                # 每改变一次  保存一下
                if local_rank == 0:
                    print('-' * 50)
                    torch.save(model.state_dict(), os.path.join(args.outf, model_path))
                counter = 0
            else:  # epoch停止条件
                counter += 1
                if counter == patience:
                    if local_rank == 0:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

                # 学习率动态调整，如果连续4个epoch没改善，调整lr
            if counter > 0 and counter % 1 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * factor, min_lr)
                if new_lr < current_lr:
                    optimizer.param_groups[0]['lr'] = new_lr
                    if local_rank == 0:
                        print(f'Learning rate reduced to {new_lr:.6f}.')
    if local_rank == 0:
        torch.save(best_model_state, os.path.join(args.best_model, f'model.pth'))

    a_tra_loss = np.mean(avg_train_loss)
    a_val_loss = np.mean(avg_validation_loss)
    if local_rank == 0:
        print(f"Average train loss: {a_tra_loss:.4f}, Average val loss: {a_val_loss:.4f}")
    dist.destroy_process_group()


if __name__ == '__main__':
    factor = 0.5
    min_lr = 1e-6
    args = prepare()
    main(args)
    local_rank = int(os.environ['LOCAL_RANK'])



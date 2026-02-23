# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 19:30:29 2025

@author: sinam
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import os
from torch.utils.data import DataLoader

import transforms
from models import cnn, PETransformerModel, DCM, FC
from timematch_utils.train_utils import bool_flag
from utils import _eval_perf, CropMappingDataset,_collate_fn
import argparse
import os
import random
import torch.nn as nn
from timematch_utils import label_utils
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

from dataset import PixelSetData, create_evaluation_loaders
from transforms import (
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
    AddPixelLabels
)


np.random.seed(10)
torch.manual_seed(10)



def args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_save_dir", type=str, default='Pretrained_USA')
    parser.add_argument("--backbone_network", type=str, choices=['CNN', 'LSTM','Transformer'])
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=50)
    # parser.add_argument("--gpu", type=list, default=[0])
    parser.add_argument("--gpu", type=list, default=[0,1,2,3])

    # 以下都是timematch
    parser.add_argument(
        "--num_workers", default=2, type=int, help="Number of workers"
    )
    parser.add_argument("--balance_source", type=bool_flag, default=True, help='class balanced batches for source')
    parser.add_argument('--num_pixels', default=4096, type=int, help='Number of pixels to sample from the input sample')
    parser.add_argument('--seq_length', default=30, type=int,
                        help='Number of time steps to sample from the input sample')
    # 数据路径与域
    parser.add_argument('--data_root', default='/data/user/DBL/timematch_data', type=str,
                        help='Path to datasets root directory')
    # parser.add_argument('--data_root', default='/mnt/d/All_Documents/documents/ViT/dataset/timematch', type=str,
    #                     help='Path to datasets root directory')
    parser.add_argument('--source', default='france/30TXT/2017', type=str)
    parser.add_argument('--target', default='france/30TXT/2017', type=str)
    # 类别处理
    parser.add_argument('--combine_spring_and_winter', action='store_true')
    # 数据划分
    parser.add_argument('--num_folds', default=3, type=int)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--test_ratio", default=0.2, type=float)
    # 评估
    parser.add_argument('--sample_pixels_val', action='store_true')  # 布尔型开关参数（flag），它不需要传值，只需在命令行中出现或不出现该选项

    return parser.parse_args()


class TupleDataset(data.Dataset):
    def __init__(self, dataset1, dataset2):
        super().__init__()
        self.weak = dataset1
        self.strong = dataset2
        assert len(dataset1) == len(dataset2)
        self.len = len(dataset1)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.weak[index], self.strong[index])

def get_data_loaders(splits, config, balance_source=True):

    strong_aug = transforms.Compose([
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            Normalize(),
            ToTensor(),
            AddPixelLabels()
    ])

    source_dataset = PixelSetData(config.data_root, config.source,
            config.classes, strong_aug,
            indices=splits[config.source]['train'],)

    if balance_source:
        source_labels = source_dataset.get_labels()
        freq = Counter(source_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_labels]
        sampler = WeightedRandomSampler(source_weights, len(source_labels))
        print("using balanced loader for source")
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=config.batch_size,
            drop_last=True,
        )
    else:
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
    print(f'size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)')

    return source_loader

def create_train_val_test_folds(datasets, num_folds, num_indices, val_ratio=0.1, test_ratio=0.2):
    folds = []
    for _ in range(num_folds):
        splits = {}
        for dataset in datasets:
            if type(num_indices) == dict:
                indices = list(range(num_indices[dataset]))
            else:
                indices = list(range(num_indices))
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val

            random.shuffle(indices)

            train_indices = set(indices[:n_train])
            val_indices = set(indices[n_train:n_train + n_val])
            test_indices = set(indices[-n_test:])
            assert set.intersection(train_indices, val_indices, test_indices) == set()
            assert len(train_indices) + len(val_indices) + len(test_indices) == n

            splits[dataset] = {'train': train_indices, 'val': val_indices, 'test': test_indices}
        folds.append(splits)
    return folds




if __name__ == "__main__":


    cfg = args()
    source_name = cfg.source.replace('/', '_')

    random.seed(10)
    config = cfg
    source_classes = label_utils.get_classes(cfg.source.split('/')[0],
                                             combine_spring_and_winter=cfg.combine_spring_and_winter)
    source_data = PixelSetData(cfg.data_root, cfg.source, source_classes)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[i] for i in labels[counts >= 200]]
    print('Using classes:', source_classes)
    cfg.classes = source_classes
    cfg.num_classes = len(source_classes)  # 可以覆盖该参数的默认设置
    # Randomly assign parcels to train/val/test
    indices = {config.source: len(source_data)}
    folds = create_train_val_test_folds([config.source], config.num_folds, indices, config.val_ratio,
                                        config.test_ratio)

    if cfg.backbone_network=="CNN":
        backbone=cnn()
        fc=FC(input_dim=1024)
    elif cfg.backbone_network=="Transformer":
        backbone=PETransformerModel()
        fc=FC(input_dim=64)
    elif cfg.backbone_network=="LSTM":
        backbone=DCM()
        fc=FC(input_dim=512)
    device = torch.device(f'cuda:{cfg.gpu[0]}' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    fc = fc.to(device)

    backbone = torch.nn.DataParallel(backbone, device_ids=cfg.gpu)
    fc = torch.nn.DataParallel(fc, device_ids=cfg.gpu)
    total_params = sum(p.numel() for p in backbone.parameters())+sum(p.numel() for p in fc.parameters())
    print("Total number of parameters: ", total_params)



    optimizer = optim.Adam(list(backbone.parameters())+list(fc.parameters()), lr=0.0001)

    criterion= nn.CrossEntropyLoss()





    best_mF1s=0
    for fold_num, splits in enumerate(folds):
        print(f'Starting fold {fold_num}...')

        config.fold_num = fold_num

        sample_pixels_val = config.sample_pixels_val
        val_loader, test_loader = create_evaluation_loaders(config.source, splits, config, sample_pixels_val)
        source_loader = get_data_loaders(splits, config, config.balance_source)
        for epoch in range(cfg.epochs):
            log = []
            backbone.train()
            fc.train()
            for i, batch in enumerate(tqdm(source_loader, desc="Processing batches")):
                # print(f"{epoch: }",i, "/", len(source_loader))
                xt_train_batch = batch["pixels"].to(device)
                B,T,C,N = xt_train_batch.shape
                xt_train_batch = xt_train_batch.permute(0, 3, 2, 1).reshape(-1,C,T)
                # print("shape:", xt_train_batch.shape)#shape: torch.Size([40960, 10, 30])
                yt_train_batch = batch["pixel_labels"].reshape(-1).to(device)
                # print("shape:", yt_train_batch.shape)#shape: torch.Size([40960])

                optimizer.zero_grad()
                outputs = backbone(xt_train_batch)
                outputs = fc(outputs) # 这里的fc即mlp映射类别，类别数硬编码到model文件FC类中
                # 假设 labels 是你的真实标签张量（shape: [B] 或 [B*N]）
                # print("Unique labels:", torch.unique(yt_train_batch))
                # print("Min label:", yt_train_batch.min().item(), "Max label:", yt_train_batch.max().item())
                assert yt_train_batch.min() >= 0 and yt_train_batch.max() < cfg.num_classes, "Label out of range!"
                loss = criterion(outputs, yt_train_batch)
                loss.backward()
                optimizer.step()

            _, acc_train, _ = _eval_perf(source_loader, backbone, fc, device)
            F1s, acc, mF1s = _eval_perf(val_loader, backbone, fc, device)

            log.append({
                'fold': fold_num,
                'epoch': epoch,
                'acc_train': acc_train,
                'val_acc': acc,
                'val_mF1s': mF1s,
                'val_F1s_per_class': str(F1s)  # F1s 是数组，转成字符串便于存储；也可单独展开列
            })
            os.makedirs(cfg.pretrained_save_dir, exist_ok=True)
            if mF1s > best_mF1s:
                best_mF1s = mF1s
                backbone_svedir = os.path.join(cfg.pretrained_save_dir, 'backbone_'+cfg.backbone_network )
                fc_svedir = os.path.join(cfg.pretrained_save_dir, 'fc_'+cfg.backbone_network )
                os.makedirs(backbone_svedir,exist_ok=True)
                os.makedirs(fc_svedir,exist_ok=True)
                torch.save(backbone.state_dict(), os.path.join(backbone_svedir , f'site_{source_name}.pth'))
                torch.save(fc.state_dict(), os.path.join(fc_svedir, f'site_{source_name}.pth'))

            print(epoch,"acc_train:",acc_train)
            print(epoch,"mF1s_val:",mF1s)
            df = pd.DataFrame(log)
            csv_path = os.path.join(cfg.pretrained_save_dir, f'training_log_{source_name}_{cfg.backbone_network}.csv')
            if  os.path.exists(csv_path):
                df.to_csv(csv_path,
                header=False,
                mode='a',           # 追加模式
                index=False)
            else :
                df.to_csv(csv_path, index=False)
            print(f"Training log saved to: {csv_path}")
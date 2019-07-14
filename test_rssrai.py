import os
import shutil
import glob
import numpy as np
from torch.utils.data import DataLoader

from dataloaders.datasets.rssrai_test import RssraiTestSet
from modeling.deeplab import *

from tqdm import tqdm
import pandas as pd
import torch

from mypath import Path
from utils.load_parameter import load_parameter

BATCH_SIZE = 80
NUM_WORKERS = 8


def eval_epoch(model, valid_set, device):
    # 加载到GPU
    model.cuda(device)
    model.eval()
    list = []
    for idx in tqdm(range(len(valid_set))):
        pics, _ = valid_set[idx]
        pics = torch.stack(pics)
        inputs = pics.cuda(device)
        pred = model(inputs)
        pred = torch.argmax(pred.mean(dim=0))
        list.append(pred.cpu().item())
    return list


def submit(model, model_name, device):
    rssrai_val = RssraiTestSet()
    dataloader = DataLoader(rssrai_val, batch_size=1, num_workers=8)
    model = model.cuda()
    model.eval()
    tbar = tqdm(dataloader, desc='\r')
    for i, sample in enumerate(tbar):
        image = sample["image"].cuda()
        name = sample["name"]
        with torch.no_grad():
            output = model(image)
            print(name)
            print(output)
            print(output.size())
        break


def test():
    pass

    # test = False
    # if test:
    #     train_csv_url = INPUT_PATH + '/train_labels.csv'
    #     data = pd.read_csv(train_csv_url)
    #     train_path = INPUT_PATH + '/train/'
    #     _, vd = train_test_split(data, test_size=0.1, random_state=123)
    #     valid_set = TTA_data_set(vd, train_path, tta_times=9)
    #     valid_len = vd.count()["id"]
    #
    #     # 加载模型
    #     # resnet18
    #     # densenet201
    #     # pnasnet5large
    #     model = densenet201(num_classes=2, pretrained=False)
    #     model_name = "densenet201"
    #     # 模型参数加载
    #     model = load_parameter(model,
    #                            model_name,
    #                            type='pre_model',
    #                            pre_model='models_weight/MyWeight/' +
    #                                      '2019-03-23--15:32:09/' +
    #                                      '2019-03-24--03:02:37--densenet201--105--Loss--0.0722--Acc--0.9769.pth')
    #
    #     model.cuda(device)
    #     model.eval()
    #
    #     # 损失函数
    #     criterion = torch.nn.CrossEntropyLoss().cuda(device)
    #     # 评估
    #     valid_acc = 0
    #     for idx in tqdm(range(valid_len)):
    #         inputs, label = valid_set[idx]
    #         # 10个
    #         inputs = torch.stack(inputs).cuda(device)
    #         # 1个
    #         pred = model(inputs)
    #         pred = torch.argmax(pred.mean(dim=0))
    #         valid_acc += int(pred.cpu().item() == label)
    #         # print(pred.cpu().item(),"==",label)
    #     valid_acc /= valid_len
    #     print(f"valid_acc: {valid_acc}")
    #     # valid_acc: 0.9777292973366057


def main():
    # 加载数据
    device = 0

    # 加载模型

    # Define network
    model = DeepLab(num_classes=16,
                    backbone="resnet_4c",
                    output_stride=16,
                    sync_bn=True,
                    freeze_bn=True)
    model_name = "deepLabV3Plus"

    # 模型参数加载
    model = load_parameter(model
                           , type="pre_model"
                           ,
                           pre_model="/home/arron/Documents/arron/pytorch-deeplab-xception/run/rssrai/deeplab-resnet_4c/model_best.pth.tar")

    submit(model, model_name, device)


if __name__ == "__main__":
    main()

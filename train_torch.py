from tensorflow.python.keras.applications import imagenet_utils
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets,models,transforms
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import cv2
import argparse
import random
from collections.abc import Iterable
from sklearn.utils import shuffle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='datasets/data', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7,
                        help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=10,
                        help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--margin_1', type=float, default=0.07,
                        help='Rank regularization margin. Details described in the paper.0.15-->1024bs')
    parser.add_argument('--margin_2', type=float, default=0.2,
                        help='Relabeling margin. Details described in the paper.')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='Batch size.')

    parser.add_argument('--optimizer', type=str,
                        default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9,
                        type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=70,
                        help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float,
                        default=0, help='Drop out rate.')
    return parser.parse_args()

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
#todo 读取数据
class MyDataSet(Dataset):
    def __init__(self,data_path,phase='train',transform = None,basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path
        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        dataset = pd.read_csv(os.path.join(self.data_path, 'trainlabel.txt'), sep=' ', header=None)  # 标签数据
        dataset = shuffle(dataset)  # 打乱顺序
        if phase == 'train':
            file_names = dataset.iloc[:8000, NAME_COLUMN].values
            # 0:正常 1:颠倒
            self.label = dataset.iloc[:8000, LABEL_COLUMN].values
        else:
            file_names = dataset.iloc[8000:, NAME_COLUMN].values
            # 0:正常 1:颠倒
            self.label = dataset.iloc[8000:, LABEL_COLUMN].values

        self.file_paths = []
        for f in file_names:
            path = os.path.join(self.data_path, f)
            self.file_paths.append(path)
        self.basic_aug=basic_aug
        self.aug_func=[imagenet_utils.flip_image,imagenet_utils.add_gaussion_noise]

    def __len__(self):
        # 图片数量
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]  # 第几张图片
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = int(self.label[idx])
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)  # 随机进行flip水平翻转图片0, 随机抖动1

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx

class MobileNetV2Feature(nn.Module):
    def __init__(self, num_classes=2,drop_rate=0):
        super(MobileNetV2Feature,self).__init__()
        self.drop_rate = drop_rate
        net = models.mobilenet_v2(True,width_mult=0.25)
        net.classifier=nn.Sequential()  # 将分类层置空
        self.features=net   # 保留特征层
        self.classifier = nn.Sequential(
            nn.Linear(1280,1000),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(1000,num_classes),
        )
        def forward(self,x):
            x = self.features(x)
            x = x.view(x.size(0),-1)  # 展平
            x = self.classifier(x)
            return x

# 定义两个函数，一个可以冻住features层，只训练FC层，另一个把features层解冻，训练所有参数
def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


def run_training():
    args = parse_args()
    net = MobileNetV2Feature(drop_rate=args.drop_rate)  


    data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
    transforms.RandomErasing(scale=(0.02, 0.25))])

    train_dataset = MyDataSet(
        args.raf_path, phase='train', transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])])
    val_dataset = MyDataSet(args.raf_path, phase='test',
                             transform=data_transforms_val)
    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)
    params = net.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported.")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    net = net.to(device)

    # 冻结 features层
    freeze_by_names(net, ('features'))

    # 解冻features层
    # unfreeze_by_names(net, ('features'))
    criterion = nn.CrossEntropyLoss()
    margin_1 = args.margin_1
    margin_2 = args.margin_2
    beta = args.beta
    modelSaved = False  # 模型是否保存过

    for i in range(1, args.epochs + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        net.train()
        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
            batch_sz = imgs.size(0)
            iter_cnt += 1
            tops = int(batch_sz * beta)
            optimizer.zero_grad()
            imgs = imgs.to(device)
            outputs = net(imgs)

            targets = np.array(targets)
            targets = torch.from_numpy(targets)
            targets = targets.to(device)

            loss = criterion(outputs, targets) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # 使用Top5分类
            maxk = max((1,5))
            label_resize = targets.view(-1, 1)
            _, predicted = outputs.topk(maxk, 1, True, True)
            total += targets.size(0)
            correct += torch.eq(predicted, label_resize).cpu().sum().float().item()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
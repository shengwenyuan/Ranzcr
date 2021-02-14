import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import os
import cv2
import copy
import time
import PIL
import torch.utils.data as Data
from torchvision import transforms, models
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.vision import VisionDataset

class RanzcrDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,):
        super(RanzcrDataset, self).__init__(root, transform=transform)    
        self.train_data = pd.read_csv(os.path.join(self.root, 'datasets',  "train.csv"),  header=0)
        self.train_data = self.train_data.values

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        fn = self.train_data[index, 0] + '.jpg'
        X = PIL.Image.open(os.path.join(self.root, 'datasets/train', fn))
        if self.transform is not None:
            X = self.transform(X)
        target = self.train_data[index, 1:-1]
        target_tensor = torch.tensor(target.tolist())
        if self.target_transform is not None:
            target_tensor = self.target_transform(target)

        return X, target_tensor

    def __len__(self) -> int:
        return len(self.train_data)

    def extra_repr(self) -> str:
            lines = ["Target type: {target_type}", "Split: {split}"]
            return '\n'.join(lines).format(**self.__dict__)

train_data_transforms=transforms.Compose([
    transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]),
])

train_target_transforms=transforms.Compose([
    None,
])

dataset = RanzcrDataset('/home/swy/kaggle', transform=train_data_transforms)
train_loader = Data.DataLoader(dataset, batch_size = 16, shuffle = False, num_workers = 4)


class RanZcrNet(nn.Module):
    def __init__(self):
        super(RanZcrNet, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, bias=False),
            )
        self.resnet= models.resnet152(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 10),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        x=self.conv1(x)
        x=self.resnet(x)
        output=self.classifier(x)
        return output

def train_model(model, traindataloader, targetCol, train_rate, criterion, optimizer, num_epochs = 30):
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num*train_rate)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model.to('cuda')

    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0
        val_corrects = 0
        val_num = 0
        for step, (images, labels) in enumerate(traindataloader):
            b_x = images.to('cuda')
            b_y = labels[:, targetCol].to('cuda')
            if step < train_batch_num:
                model.train()
                output = model(b_x)
                pre_lab = torch.argmax(output,1)
                loss = criterion(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)
            else:
                model.eval()
                output = model(b_x)
                pre_lab = torch.argmax(output,1)
                loss = criterion(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(data = {'epoch' : range(num_epochs), 
        'train_loss_all' : train_loss_all, 
        'val_loss_all' :   val_loss_all, 
        'train_acc_all' : train_acc_all, 
        'val_acc_all' : val_acc_all})
    b_x.cpu()
    b_y.cpu()
    output.cpu()
    pre_lab.cpu()
    loss.cpu()
    model.cpu()

    return model, train_process

for i in [2, 8]:   #label2 works but label8 does not
    torch.cuda.init()
    targetCol=i
    net=RanZcrNet()

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    model, train_process = train_model(net, train_loader, targetCol, 0.8, criterion, opt, num_epochs=61)
    train_process.to_csv('train_process_resnet152_col'+str(targetCol))
    torch.save(model, 'model_resnet152_col'+str(targetCol))
    torch.cuda.empty_cache()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from part2.CNN_net import CNNNet

EPOCH = 10
BATCH_SIZE = 100
LR = 0.001
train_ratio = 0.9
torch.manual_seed(1)  # 保证可以复现


if __name__ == "__main__":


    #####################################################
    # 数据处理和初始化
    # 训练集和验证机的划分
    # transform = transforms.Compose([transforms.ToTensor(), 					# 把灰度范围从[0,255]变换到[0,1]
    #                                 transforms.Normalize((0.5), (0.5))])    # 把[0,1]变换到[-1,1]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5),(0.5))    # 归一化，加快收敛速度
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5),(0.5))    # 归一化，加快收敛速度
        ])
    }

    dataset = torchvision.datasets.ImageFolder("../train/train",transform=data_transforms["train"])  # 读取数据集


    # 划分训练集和验证集合
    train_dataset,val_dataset = data.random_split(dataset,[int(train_ratio*len(dataset)),len(dataset)-int(train_ratio*len(dataset))])
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset,batch_size=BATCH_SIZE,shuffle=False)


    cnnnet = CNNNet()
    print("网络结构如下：")
    print(cnnnet)
    optimizer = optim.Adam(cnnnet.parameters(),lr = LR)
    criterion = nn.CrossEntropyLoss()   # CE Loss
    loss_list = []
    start = time.time()    # 记录训练开始的时间
    for epoch in range(EPOCH):
        cnnnet.train()
        for i, x_y in tqdm(list(enumerate(train_loader))):
            batch_x = Variable(x_y[0])
            batch_y = Variable(x_y[1])
            output = cnnnet.forward(batch_x)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Loss: {:.6f}'.format(epoch, loss.item()))
        loss_list.append(loss.item())  # 用于绘图

    print('\nFinished Training! Total cost time: ', time.time()-start)

    cnnnet.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = cnnnet(data)
            loss = criterion(output, target)
            val_loss += loss.item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            val_loss /= len(val_loader.dataset)

    print('\nval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    torch.save(cnnnet,"cnn-g-graph.pth")

    # plt.plot(list(range(EPOCH)),loss_list)
    # plt.title("损失-轮数变化图")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")



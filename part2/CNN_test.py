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
from part2.CNN_net import CNNNet


if __name__ == "__main__":

    # model = CNNNet()
    #
    # is_load = True
    # model_path = './cnn-g.pth'
    # if is_load:
    #     params = torch.load(model_path)
    #     model.load_state_dict(params, strict=True)

    model = torch.load("cnn-g.pth")

    BATCH_SIZE = 100

    data_transforms = {
        'val': transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5),(0.5))    # 归一化，加快收敛速度
        ])
    }


    # valid_set = torchvision.datasets.ImageFolder("../train/train",transform=data_transforms["val"])  # 读取数据集
    valid_set = torchvision.datasets.ImageFolder(r"C:\Users\84663\Desktop\test_data",transform=data_transforms["val"])  # 读取数据集
    train_ratio = 0   # 全是验证集

    # 划分训练集和验证集合
    # train_dataset,val_dataset = data.random_split(dataset,[int(train_ratio*len(dataset)),len(dataset)-int(train_ratio*len(dataset))])

    val_loader = data.DataLoader(dataset=valid_set,batch_size=BATCH_SIZE,shuffle=False)

    # cnnnet = torch.load("cnn-g.pth")  # 导出模型

    loss_func = nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model.forward(data)
            loss = loss_func(output, target)
            val_loss += loss.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            val_loss /= len(val_loader.dataset)
    # print("djkashdfsahdlkashfkldahfklsadhfkhasdklfhasdklfhasdklhfklasdhfklsadhfklasdhfklsadhfklashfkasdhfklasdhflkasd")
    print('\nval set: Average loss: {:.8f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


import torch
import torchvision.models as models
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

resnet50 = models.resnet50(pretrained=True)  #调用预训练模型
loss_function = nn.CrossEntropyLoss()  #定义损失函数
optimizer = optim.SGD(resnet50.parameters(), lr=0.1)

resnet50.fc = nn.Linear(2048,10)  # 修改模型尺寸
resnet50.conv1 = nn.Conv2d(1, 64,kernel_size=5, stride=2, padding=3, bias=False)
print(resnet50)

######################## 导入数据##########################
# 预处理
my_transform = transforms.Compose([
        # transforms.Resize((224, 224)),   # 修改图片尺寸以适应resnet的输入
        # transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081)),
        # transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
    ])

# 训练集
train_file = datasets.MNIST(
    root='data',
    train=True,
    transform=my_transform
)
# 测试集
test_file = datasets.MNIST(
    root='data',
    train=False,
    transform=my_transform
)

train_loader = torch.utils.data.DataLoader(train_file,batch_size=64)
test_loader = torch.utils.data.DataLoader(test_file,batch_size=64)

#########################训练函数###########################
def train(model,train_loader,loss_func,optim,epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = loss_func(output,target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


#########################测试函数###########################
def test(model, test_loader,loss_func):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # if batch_idx == 10000:
            # break
            output = model(data)
            test_loss += loss_func(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() #compare prediction and lable
            test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__=="__main__":
    for epoch in range(2):
        train(resnet50,train_loader,loss_function,optimizer,epoch)
        test(resnet50,test_loader,loss_function)
    torch.save(resnet50.state_dict(),"minist_resnet.pt")
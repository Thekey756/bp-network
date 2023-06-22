import torch
import torchvision.models as models
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 预处理
my_transform = transforms.Compose([
        # transforms.Resize((224, 224)),   # 修改图片尺寸以适应resnet的输入
        # transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081)),
        # transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
    ])
# 测试集
test_file = datasets.MNIST(
    root='data',
    train=False,
    transform=my_transform
)

test_loader = torch.utils.data.DataLoader(test_file,batch_size=64)

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
    model_path = "./minist_resnet_best.pt"
    # model = torch.load(model_path)
    model = models.resnet50()
    model.fc = nn.Linear(2048, 10)  # 修改模型尺寸
    model.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=3, bias=False)

    model.load_state_dict(torch.load("minist_resnet_best.pt"))
    loss_function = nn.CrossEntropyLoss()
    test(model, test_loader, loss_function)
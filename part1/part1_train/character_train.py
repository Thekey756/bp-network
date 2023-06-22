from mytorch.nn.functional import Tanx,Sigmoid,Softmax,Activate
from mytorch.nn.net import Linear,character_net,basenet
from mytorch.utils import Dataset,DataLoader
import numpy as np
from mytorch.nn.functional import MSELoss,CELoss
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm,trange

if __name__ == '__main__':
    dataloader = DataLoader("train",train_ratio=0.9,shuffle=True,expand=4)
    train_data = dataloader.data["train"]
    validation_data = dataloader.data["validation"]
    train_len = len(train_data)
    valid_len = len(validation_data)


    epoch = 15
    characterNet = character_net()
    # 由于DataLoader对数据的处理，第一层的输入需要是28*28
    # 其他的参数可以调整

    layer1 = Linear(28*28,32*16,Sigmoid())
    layer2 = Linear(32*16,16*16,Sigmoid())
    layer3 = Linear(16*16,16*8,Sigmoid())
    layer4 = Linear(16*8,12,Softmax())

    # layer1 = Linear(28*28, 28*16, Sigmoid())
    # layer2 = Linear(28*16,16*16,Sigmoid())
    # layer3 = Linear(16*16,16*8,Sigmoid())
    # layer4 = Linear(16*8,12,Softmax())

    # layer1 = Linear(28 * 28, 24 * 20, Sigmoid())
    # layer2 = Linear(24 * 20, 16 * 16, Sigmoid())
    # layer3 = Linear(16 * 16, 14 * 12, Sigmoid())
    # layer4 = Linear(14 * 12, 12, Softmax())

    # layer1 = Linear(28 * 28, 24 * 24, Sigmoid())
    # layer2 = Linear(24 * 24, 24 * 24, Sigmoid())
    # layer3 = Linear(24 * 24, 16 * 16, Sigmoid())
    # layer4 = Linear(16 * 16, 12, Softmax())

    # layer1 = Linear(28 * 28, 22 * 20, Sigmoid())
    # layer2 = Linear(22 * 20, 20 * 16, Sigmoid())
    # layer3 = Linear(20 * 16, 12 * 8, Sigmoid())
    # layer4 = Linear(12 * 8, 12, Softmax())             # 到达90%了

    # layer1 = Linear(28 * 28, 22 * 18, Sigmoid())
    # layer2 = Linear(22 * 18, 18 * 14, Sigmoid())
    # layer3 = Linear(18 * 14, 12 * 8, Sigmoid())
    # layer4 = Linear(12 * 8, 12, Softmax())

    # layer1 = Linear(28 * 28, 26 * 18, Sigmoid())
    # layer2 = Linear(26 * 18, 20 * 16, Sigmoid())
    # layer3 = Linear(20 * 16, 12 * 8, Sigmoid())
    # layer4 = Linear(12 * 8, 12, Softmax())
    characterNet.setup(0.01,layer1,layer2,layer3,layer4)
    loss = CELoss()

    loss_list = []
    t_acc_list = []
    v_acc_list = []
    for i in range(epoch):
        np.random.shuffle(train_data)
        train_corret = 0
        ls = 0
        train_data_ = tqdm(train_data)
        for x, y in train_data_:
            train_data_.set_description("epoch:%d" % i)
            y_pred_value = characterNet.forward(x)
            ls += loss.loss_calculate(characterNet, y)
            loss.backprop()
            y_pred = np.argmax(y_pred_value)
            y_real = np.argmax(y)
            if y_pred == y_real:
                train_corret += 1
        # print("第{}轮训练的误差是{}".format(i,ls))
        train_accuracy = train_corret / train_len
        t_acc_list.append(train_accuracy)
        loss_list.append(ls)

        valid_corret = 0
        valid_ls = 0
        for x, y in validation_data:
            y_pred_value = characterNet.forward(x)
            valid_ls += loss.loss_calculate(characterNet, y)
            y_pred = np.argmax(y_pred_value)
            y_real = np.argmax(y)
            if y_pred == y_real:
                valid_corret += 1
        valid_accuracy = valid_corret / valid_len
        v_acc_list.append(valid_accuracy)

        print("epoch:{},训练集准确率:{}, 验证集准确率:{}, 训练集loss:{}, 验证集loss:{}".format(i, train_accuracy, valid_accuracy, ls, valid_ls))

    f = open('models/charNet_nu_51', 'wb')
    pickle.dump(characterNet, f)
    f.close()
    print("模型存储完毕")

    plt.subplot(2,2,1)
    plt.plot(list(range(epoch)), loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss in Each Epoch")
    # plt.show()
    plt.subplot(2,2,2)
    plt.plot(list(range(epoch)), t_acc_list)
    plt.xlabel("epoch")
    plt.ylabel("train_accuracy")
    plt.title("accuracy in train set")
    # plt.show()
    plt.subplot(2,2,3)
    plt.plot(list(range(epoch)), v_acc_list)
    plt.xlabel("epoch")
    plt.ylabel("valid_accuracy")
    plt.title("accuracy in validation set")
    plt.show()






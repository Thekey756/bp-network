from mytorch.nn.functional import Tanx,Sigmoid,Softmax,Activate
from mytorch.nn.net import Linear,sin_net
import numpy as np
from mytorch.nn.functional import MSELoss
import pickle
from tqdm import tqdm,trange

if __name__ == "__main__":
    # print("==========================") # 27个=
    sinNet = sin_net()
    # sigmoid = Sigmoid()
    activate = Activate()
    layer1 = Linear(1,7, Sigmoid())
    layer2 = Linear(7,5,Sigmoid())
    layer3 = Linear(5,1,activate)

    sinNet.setup(0.001,layer1,layer2,layer3)

    n_feature = 1
    n_count = 4000
    epoch = 500

    x = (2 * np.random.rand(n_count, n_feature) - 1) * np.pi #生成一个（4000，1）的数组，元素大小在正负Π之间
    x_list = []
    for i in x:
        x_list.append(np.reshape(i, (n_feature, 1)))    # [[x],[y],[z],···]
    x_list = np.array(x_list)
    y_list = np.sin(x_list)  # label
    loss = MSELoss()
    # print(x_list.shape)
    # exit()

    for i in tqdm(range(epoch)):
        ls = 0
        for j in range(n_count):
            X = x_list[j]
            y = y_list[j]

            y_pred = sinNet.forward(X)
            ls += loss.loss_calculate(sinNet,y)

            loss.backprop()
        average_loss = ls / n_count
        print("general loss = {}，average loss = {}".format(ls,average_loss))

    f = open('models/sinNet', 'wb')
    pickle.dump(sinNet, f)
    f.close()
    print("模型存储完毕")




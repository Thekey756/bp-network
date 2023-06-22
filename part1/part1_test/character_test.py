from mytorch.nn.functional import Tanx,Sigmoid,Softmax,Activate
from mytorch.nn.net import Linear,sin_net,basenet
from mytorch.utils import Dataset,DataLoader
import numpy as np
from mytorch.nn.functional import MSELoss,CELoss
import pickle
from tqdm import tqdm,trange
from mytorch.nn.net import character_net




if __name__ == "__main__":
    # imageLoader = DataLoader("../part1_train/train", train_ratio=0.9)
    imageLoader = DataLoader(r"C:\Users\84663\Desktop\test_data", train_ratio=0)
    # C:\Users\84663\Desktop\train
    validation_data_list = imageLoader.data["validation"]

    # imageLoader2 = DataLoader("../part1_train/train", train_ratio=0)
    # test = imageLoader2["validation"]

    with open("../part1_train/models/charNet_nu_51", 'rb') as f:
        tryNet = pickle.load(f)
    # with open("../part1_train/models/charNet(1)", 'rb') as f:
    #     tryNet = pickle.load(f)

    total = len(validation_data_list)
    acc = 0
    for x, y in validation_data_list:

        y_pred_value = tryNet.forward(x)
        y_pred = np.argmax(y_pred_value)
        y_real = np.argmax(y)

        if y_pred == y_real:
            acc += 1

    print("准确率为:", acc/total)



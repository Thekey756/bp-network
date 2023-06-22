# from mytorch.nn.functional import Tanx,Sigmoid,Softmax,Activate
# from mytorch.nn.net import Linear,sin_net,basenet
# from mytorch.utils import Dataset,DataLoader
# import numpy as np
# from mytorch.nn.functional import MSELoss,CELoss
# import pickle
# from tqdm import tqdm,trange
#
# class sin_net(basenet):
#     def __init__(self):
#         self.lr = 0.005
#         self.Layer1 = None
#         self.Layer2 = None
#
#
#     def setup(self,lr,layer1,layer2):
#         self.lr = lr
#         self.Layer1 = layer1
#         self.Layer2 = layer2
#         self.connect(self.Layer1,self.Layer2)
#
#
#     def forward(self, X):
#         X = self.Layer1.forward(X)
#         X = self.Layer2.forward(X)
#         return X
#
# sinNet = sin_net()
# # sigmoid = Sigmoid()
# # activate = Activate()
# layer1 = Linear(1,5, Sigmoid())
# layer2 = Linear(5,1,Activate())
# sinNet.setup(0.01,layer1,layer2)
#
# x_ = np.array([[[1]],[[2]],[[3]],[[4]]])
# y_ = x_*x_
# # print(y_)
# # print(x_.shape)
# epoch = 1000
# loss = MSELoss()
# y_pred = sinNet.forward([[1]])
# print(y_pred)
# for i in range(epoch):
#     ls = 0
#     for j in range(2):
#
#         X = x_[j]
#         y = y_[j]
#
#
#         y_pred = sinNet.forward(X)
#
#         ls += loss.loss_calculate(sinNet,y)
#         print(sinNet.last_layer.a)
#         loss.backprop()
#     # print(ls)
# y_pred = sinNet.forward([[1]])
# print(y_pred)
#
# # f = open('models/sinNet', 'wb')# # pickle.dump(sinNet, f)
# # f.close()
# # print("模型存储完毕")
import numpy as np
from matplotlib.pyplot import plot
from matplotlib import pyplot as plt
# x = np.linspace(-10,10, num=1000)
# y =1 /(1 + np.exp(x))
# plt.plot(x,y)
# plt.show()
a = 0xAABBCCDD
# print(a)
b = 0x1d824
c = 0xffff
print(c - b)




# 0x2d70252d-0x252d7025-0x70252d70
# a = 0x2d70252d
# b = 0x252d7025
# c = 0x70252d70
# a1 = 0x2d
# a2 = 0x70
# a3 = 0x25
# a4 = 0x2d
# print(chr(a1)+chr(a2)+chr(a3)+chr(a4),end='')
# b1 = 0x25
# b2 = 0x2d
# b3 = 0x70
# b4 = 0x25
# print(chr(b1)+chr(b2)+chr(b3)+chr(b4),end='')
# c1 = 0x70
# c2 = 0x25
# c3 = 0x2d
# c4 = 0x70
# print(chr(c1)+chr(c2)+chr(c3)+chr(c4),end='')

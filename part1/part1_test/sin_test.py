import numpy as np
import pickle
from matplotlib import pyplot as plt
from mytorch.nn.net import sin_net


n_count = 1000

x = np.linspace(-np.pi, np.pi, num=n_count)
x_list = []
for x_ in x:
    x_list.append(np.array([[x_]]))
x_list = np.array(x_list)


with open('../part1_train/models/sinNet', 'rb') as f:
    sinNet = pickle.load(f)


y_list = []
for i in range(n_count):
    y_list.append(sinNet.forward(x_list[i]))
y_list = np.array(y_list)

x_list = np.squeeze(x_list)
y_list = np.squeeze(y_list)

plt.axis([-5, 5, -2, 2])
plt.subplot(2,1,1)
plt.plot(x_list,np.sin(x_list))
plt.xlabel('x')
plt.ylabel('sinx')
plt.title("true sin")
plt.subplot(2,1,2)
plt.plot(x_list, y_list)
plt.xlabel('x')
plt.ylabel('sinx_predict')
plt.title("predicted sin")
plt.show()

print(sinNet.forward(np.array([[np.pi/2]])))

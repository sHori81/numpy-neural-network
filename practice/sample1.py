import numpy as np
import mnist

X = mnist.download_and_parse_mnist_file(\
    r"C:\caex4\le4nn\le4nn\mnist_data\train\train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file(\
    r"C:\caex4\le4nn\le4nn\mnist_data\train\train-labels-idx1-ubyte.gz")

import matplotlib.pyplot as plt
from pylab import cm
idx = 100

#plt.imshow(X[idx], cmap=cm.Greys_r)
#plt.show()
print(X[idx].reshape(-1))
print(Y[idx])
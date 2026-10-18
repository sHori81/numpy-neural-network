import numpy as np
import matplotlib.pyplot as plt

a = np.array([[1, 2, 3], [4, 5, 6]])

def fun(x):
    return x / np.sum(x, axis=-1, keepdims=True)

print(fun(a))
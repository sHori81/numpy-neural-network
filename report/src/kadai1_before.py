import numpy as np
import mnist
import matplotlib.pyplot as plt

np.random.seed(81)

image_size = np.array([28, 28]) #画像の縦横サイズ
number_of_images = 1000
class_size = 10 #クラス数
hidden_nodes = 500 #隠れ層のノード数 
input_from_keyboard = int(input("Enter an integer between 1 and 1000: "))

image_pixels = image_size[0] * image_size[1] #画像のピクセル数
var = 1.0 / np.sqrt(image_pixels) #重みの初期化に使う分散

W1 = var * np.random.randn(image_pixels, hidden_nodes)  #入力層から隠れ層への重み
W2 = var * np.random.randn(hidden_nodes, class_size)  #隠れ層から出力層への重み
b1 = var * np.random.randn(hidden_nodes)  #隠れ層のバイアス
b2 = var * np.random.randn(class_size)  #出力層のバイアス

def input_layer(input_data):
    X = mnist.download_and_parse_mnist_file(\
    r"C:\caex4\le4nn\le4nn\mnist_data\train\train-images-idx3-ubyte.gz")
    return (X[input_data].reshape(-1))

def dense_layer(x, weights, bias):
    return np.dot(x, weights) + bias

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

vectorized_input = input_layer(input_from_keyboard)
print(vectorized_input)
hidden_layer_output = dense_layer(vectorized_input, W1, b1)
print(hidden_layer_output)
hidden_layer_activation = sigmoid(hidden_layer_output)
print(hidden_layer_activation)
output_layer_output = dense_layer(hidden_layer_activation, W2, b2)
print(output_layer_output)
output_layer_activation = softmax(output_layer_output)
print(output_layer_activation)

final_output = np.argmax(output_layer_activation)

print(final_output)
from pathlib import Path

import numpy as np
import mnist

np.random.seed(10)

DATA_DIR = Path(__file__).parent / "mnist_data" #MNISTデータの置き場所
image_size = (28, 28) #画像の縦横サイズ
image_pixels = image_size[0] * image_size[1] #画像のピクセル数
pixel_value = 255.0 #画素値の最大値
class_size = 10 #クラス数
hidden_nodes = 500 #隠れ層のノード数
batch_size = 100 #バッチサイズ


def load_mnist(kind="train"):
    #MNISTの画像とラベルを読み込む (kindは"train"か"test")
    prefix = "train" if kind == "train" else "t10k"
    X = mnist.download_and_parse_mnist_file(str(DATA_DIR / kind / f"{prefix}-images-idx3-ubyte.gz"))
    Y = mnist.download_and_parse_mnist_file(str(DATA_DIR / kind / f"{prefix}-labels-idx1-ubyte.gz"))
    return X, Y


def get_minibatch(X, Y, batch_size):
    #ランダムに選んだbatch_size枚の画像(正規化済み)と正解ラベル(one-hot)を返す
    selected_indices = np.random.choice(X.shape[0], batch_size, replace=False) #ランダムな画像を選ぶ
    x = X[selected_indices].reshape(batch_size, -1).astype(np.float64) / pixel_value #画素値を0〜1に正規化
    y = np.eye(class_size)[Y[selected_indices]] #one-hotベクトルに変換
    return x, y


def dense_layer(x, weights, bias):
    return np.dot(x, weights) + bias

def sigmoid(x):
    x = np.clip(x, -500, 500) #np.expのオーバーフローを防ぐ
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) #np.expのオーバーフローを防ぐ
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy(y_true, y_pred):
    eps = 1e-7 #log(0)を防ぐ
    return -np.sum(y_true * np.log(y_pred + eps)) / y_pred.shape[0] #バッチ内の平均


def forward(x, W1, b1, W2, b2):
    #順伝播を行い、各クラスの確率 (バッチ数, class_size) を返す
    hidden_layer_activation = sigmoid(dense_layer(x, W1, b1))
    return softmax(dense_layer(hidden_layer_activation, W2, b2))


def main():
    std1 = 1.0 / np.sqrt(image_pixels) #入力層→隠れ層の重みの初期化に使う標準偏差
    std2 = 1.0 / np.sqrt(hidden_nodes) #隠れ層→出力層の重みの初期化に使う標準偏差

    W1 = std1 * np.random.randn(image_pixels, hidden_nodes) #入力層から隠れ層への重み
    W2 = std2 * np.random.randn(hidden_nodes, class_size) #隠れ層から出力層への重み
    b1 = std1 * np.random.randn(hidden_nodes) #隠れ層のバイアス
    b2 = std2 * np.random.randn(class_size) #出力層のバイアス

    X, Y = load_mnist("train")
    x_batch, y_batch = get_minibatch(X, Y, batch_size)

    y_pred = forward(x_batch, W1, b1, W2, b2)
    print(cross_entropy(y_batch, y_pred))


if __name__ == "__main__":
    main()

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import mnist

np.random.seed(0)

DATA_DIR = Path(__file__).parent / "mnist_data" #MNISTデータの置き場所
image_size = (28, 28) #画像の縦横サイズ
image_pixels = image_size[0] * image_size[1] #画像のピクセル数
pixel_value = 255.0 #画素値の最大値
class_size = 10 #クラス数
hidden_nodes = 1000 #隠れ層のノード数
batch_size = 100 #バッチサイズ
epochs = 50 #エポック数
learning_rate = 0.01 #学習率


def load_mnist(kind="train"):
    #MNISTの画像とラベルを読み込む (kindは"train"か"test")
    prefix = "train" if kind == "train" else "t10k"
    X = mnist.download_and_parse_mnist_file(str(DATA_DIR / kind / f"{prefix}-images-idx3-ubyte.gz"))
    Y = mnist.download_and_parse_mnist_file(str(DATA_DIR / kind / f"{prefix}-labels-idx1-ubyte.gz"))
    return X, Y

def preprocess(X, Y):
   #画像を正規化し、ラベルをone-hotベクトルに変換する
   x = X.reshape(X.shape[0], -1).astype(np.float64) 
   y = np.eye(class_size)[Y] 
   return x, y

"""
def get_minibatch(X, Y, batch_size):
    #ランダムに選んだbatch_size枚の画像(正規化済み)と正解ラベル(one-hot)を返す
    selected_indices = np.random.choice(X.shape[0], batch_size, replace=False) #ランダムな画像を選ぶ
    x = X[selected_indices].reshape(batch_size, -1).astype(np.float64) / pixel_value #画素値を0〜1に正規化
    y = np.eye(class_size)[Y[selected_indices]] #one-hotベクトルに変換
    return x, y
"""

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
    output_layer_activation = softmax(dense_layer(hidden_layer_activation, W2, b2))
    return hidden_layer_activation, output_layer_activation

def backward(x, y_true, hidden_layer_activation, y_pred, W2, batch_size):
    #逆伝播を行い、勾配を返す
    
    # 出力層の誤差
    output_error = (y_pred - y_true) / batch_size  # バッチ内の平均

    # 隠れ層の誤差
    hidden_error = np.dot(output_error, W2.T) * hidden_layer_activation * (1 - hidden_layer_activation)

    # 勾配の計算
    dW2 = np.dot(hidden_layer_activation.T, output_error)
    db2 = np.sum(output_error, axis=0)
    dW1 = np.dot(x.T, hidden_error)
    db1 = np.sum(hidden_error, axis=0)

    return dW1, db1, dW2, db2

def accuracy(y_true, y_pred):
    #正解ラベルと予測ラベルの一致率を計算する
    y_true_labels = np.argmax(y_true, axis=1) #one-hotベクトルからラベルに変換
    y_pred_labels = np.argmax(y_pred, axis=1) #確率分布からラベルに変換
    return np.mean(y_true_labels == y_pred_labels) #一致率を計算

def plot_history(history):
    #損失と精度の推移をグラフに描画する
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], label="Test Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    std1 = 1.0 / np.sqrt(image_pixels) #入力層→隠れ層の重みの初期化に使う標準偏差
    std2 = 1.0 / np.sqrt(hidden_nodes) #隠れ層→出力層の重みの初期化に使う標準偏差

    W1 = std1 * np.random.randn(image_pixels, hidden_nodes) #入力層から隠れ層への重み
    W2 = std2 * np.random.randn(hidden_nodes, class_size) #隠れ層から出力層への重み
    b1 = std1 * np.random.randn(hidden_nodes) #隠れ層のバイアス
    b2 = std2 * np.random.randn(class_size) #出力層のバイアス

    x_train, y_train = preprocess(*load_mnist("train")) #訓練データの読み込みと前処理
    x_test, y_test = preprocess(*load_mnist("test")) #テストデータの読み込みと前処理

    number_of_sheets = x_train.shape[0] #訓練データの枚数

    iterations_per_epoch = number_of_sheets // batch_size #1エポックあたりのイテレーション数(N/B)

    history = {"loss": [], "train_acc": [], "test_acc": []} #損失、訓練精度、テスト精度を保持する辞書

    for epoch in range(epochs):

        perm = np.random.permutation(number_of_sheets) #訓練データのインデックスをランダムに並べ替える
        loss_sum = 0.0 #損失の合計値

        for i in range(iterations_per_epoch):
            batch_indices = perm[i * batch_size : (i + 1)* batch_size] #バッチのインデックスを取得
            x_batch = x_train[batch_indices] #バッチの入力データ
            y_batch = y_train[batch_indices] #バッチの正解ラベル

            hidden_layer_activation, y_pred = forward(x_batch, W1, b1, W2, b2) #順伝播
            loss = cross_entropy(y_batch, y_pred) #損失の計算
            loss_sum += loss #損失の合計値に加算

            dW1, db1, dW2, db2 = backward(x_batch, y_batch, hidden_layer_activation, y_pred, W2, batch_size) #逆伝播

            #パラメータの更新
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        #エポックごとの損失と精度の計算
        average_loss = loss_sum / iterations_per_epoch #1エポック内の平均損失
        train_acc = accuracy(y_train, forward(x_train, W1, b1, W2, b2)[1])#訓練データに対する精度の計算
        test_acc = accuracy(y_test, forward(x_test, W1, b1, W2, b2)[1])#テストデータに対する精度の計算

        #グラフに描画
        history["loss"].append(average_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)  

        print(f"epoch {epoch:3d}  loss {average_loss:.4f}  train_acc {train_acc:.4f}  test_acc {test_acc:.4f}")  

    plot_history(history)


if __name__ == "__main__":
    main()

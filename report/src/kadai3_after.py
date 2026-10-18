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
eval_chunk_size = 10000 #正答率を計算するときに一度に順伝播する枚数 (メモリ節約のため)


def load_mnist(kind="train"):
    #MNISTの画像とラベルを読み込む (kindは"train"か"test")
    prefix = "train" if kind == "train" else "t10k"
    X = mnist.download_and_parse_mnist_file(str(DATA_DIR / kind / f"{prefix}-images-idx3-ubyte.gz"))
    Y = mnist.download_and_parse_mnist_file(str(DATA_DIR / kind / f"{prefix}-labels-idx1-ubyte.gz"))
    return X, Y

def preprocess(X, Y):
    #画像を(枚数, 784)にし、ラベルをone-hotベクトルに変換する
    #TODO: 正規化 (/ pixel_value) をするかどうか決める
    x = X.reshape(X.shape[0], -1).astype(np.float64) / pixel_value #画素値を0〜1に正規化
    y = np.eye(class_size)[Y]
    return x, y


def init_params():
    #重みとバイアスを初期化する
    std1 = 1.0 / np.sqrt(image_pixels) #入力層→隠れ層の重みの初期化に使う標準偏差
    std2 = 1.0 / np.sqrt(hidden_nodes) #隠れ層→出力層の重みの初期化に使う標準偏差

    #乱数を使う順番 (W1→W2→b1→b2) を変えると結果が変わるので注意
    W1 = std1 * np.random.randn(image_pixels, hidden_nodes) #入力層から隠れ層への重み
    W2 = std2 * np.random.randn(hidden_nodes, class_size) #隠れ層から出力層への重み
    b1 = std1 * np.random.randn(hidden_nodes) #隠れ層のバイアス
    b2 = std2 * np.random.randn(class_size) #出力層のバイアス
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


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


def forward(x, params):
    #順伝播を行い、隠れ層の出力と各クラスの確率 (バッチ数, class_size) を返す
    hidden_layer_activation = sigmoid(dense_layer(x, params["W1"], params["b1"]))
    output_layer_activation = softmax(dense_layer(hidden_layer_activation, params["W2"], params["b2"]))
    return hidden_layer_activation, output_layer_activation

def backward(x, y_true, hidden_layer_activation, y_pred, params):
    #逆伝播を行い、各パラメータの勾配を返す
    output_error = (y_pred - y_true) / y_pred.shape[0] #出力層の誤差 (バッチ内の平均)
    hidden_error = np.dot(output_error, params["W2"].T) * hidden_layer_activation * (1 - hidden_layer_activation) #隠れ層の誤差

    return {
        "W1": np.dot(x.T, hidden_error),
        "b1": np.sum(hidden_error, axis=0),
        "W2": np.dot(hidden_layer_activation.T, output_error),
        "b2": np.sum(output_error, axis=0),
    }


def accuracy(x, y_true, params):
    #正答率を返す (一度に全部を順伝播するとメモリを大量に使うので、eval_chunk_size枚ずつ計算する)
    correct = 0
    for start in range(0, x.shape[0], eval_chunk_size):
        _, y_pred = forward(x[start:start + eval_chunk_size], params)
        correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true[start:start + eval_chunk_size], axis=1))
    return correct / x.shape[0]


def train_one_epoch(x_train, y_train, params):
    #訓練データを重複なくbatch_size枚ずつ使って1エポック学習し、1エポック内の平均損失を返す
    number_of_sheets = x_train.shape[0] #訓練データの枚数
    iterations_per_epoch = number_of_sheets // batch_size #1エポックあたりのイテレーション数(N/B)

    perm = np.random.permutation(number_of_sheets) #訓練データのインデックスをランダムに並べ替える
    loss_sum = 0.0 #損失の合計値

    for i in range(iterations_per_epoch):
        batch_indices = perm[i * batch_size:(i + 1) * batch_size] #バッチのインデックスを取得
        x_batch = x_train[batch_indices] #バッチの入力データ
        y_batch = y_train[batch_indices] #バッチの正解ラベル

        hidden_layer_activation, y_pred = forward(x_batch, params) #順伝播
        loss_sum += cross_entropy(y_batch, y_pred) #損失の合計値に加算

        grads = backward(x_batch, y_batch, hidden_layer_activation, y_pred, params) #逆伝播
        for key in params: #パラメータの更新
            params[key] -= learning_rate * grads[key]

    return loss_sum / iterations_per_epoch


def plot_history(history, file_name="history.png"):
    #損失と精度の推移をグラフに描画し、画像として保存する
    epoch_axis = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_axis, history["loss"], label="Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_axis, history["train_acc"], label="Train Accuracy")
    plt.plot(epoch_axis, history["test_acc"], label="Test Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(file_name) #ウィンドウを閉じる前に中断しても画像が残るように先に保存
    plt.show()


def main():
    params = init_params()

    x_train, y_train = preprocess(*load_mnist("train")) #訓練データの読み込みと前処理
    x_test, y_test = preprocess(*load_mnist("test")) #テストデータの読み込みと前処理

    history = {"loss": [], "train_acc": [], "test_acc": []} #損失、訓練精度、テスト精度を保持する辞書

    for epoch in range(1, epochs + 1):
        average_loss = train_one_epoch(x_train, y_train, params) #1エポック内の平均損失
        train_acc = accuracy(x_train, y_train, params) #訓練データに対する精度
        test_acc = accuracy(x_test, y_test, params) #テストデータに対する精度

        #履歴に記録
        history["loss"].append(average_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"epoch {epoch:3d}  loss {average_loss:.4f}  train_acc {train_acc:.4f}  test_acc {test_acc:.4f}")

    plot_history(history)


if __name__ == "__main__":
    main()

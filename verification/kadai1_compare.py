# 課題1のリファクタリング前後の比較（レポート 2.5.1 節・表4）を再現するプログラム
#
# report/src/kadai1_before.py と report/src/kadai1_after.py は、
# 画像番号をキーボードから1つ受け取って1枚ずつ認識するため、1000枚を試すには1000回実行する必要がある。
# どちらのプログラムも「シード値81 → W1, W2, b1, b2 の順に乱数を生成」しており、
# 入力される画像番号によらず毎回同じ重みになる。
# そこでこのプログラムでは、同じ手順で重みを1回だけ作り、画像番号 1〜1000 の1000枚をまとめて計算する。
# （1枚ずつ実行した場合と結果が一致することは、最後に元のプログラムを実際に実行して確かめる）
#
# 実行方法（リポジトリのフォルダで）：python verification/kadai1_compare.py

import subprocess
import sys
from pathlib import Path

import numpy as np
import mnist

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "mnist_data" / "train"

seed = 81 #元のプログラムと同じシード値
image_pixels = 28 * 28 #画像のピクセル数
hidden_nodes = 500 #隠れ層のノード数
class_size = 10 #クラス数
pixel_value = 255.0 #画素値の最大値
image_numbers = np.arange(1, 1001) #キーボードから入力できる画像番号 1〜1000


def make_weights(same_std):
    # 元のプログラムと同じ順番 (W1 → W2 → b1 → b2) で重みとバイアスを作る
    # same_std=True ならリファクタリング前と同じく、全層で標準偏差 1/sqrt(784) を使う
    np.random.seed(seed)
    std1 = 1.0 / np.sqrt(image_pixels)
    std2 = std1 if same_std else 1.0 / np.sqrt(hidden_nodes)
    W1 = std1 * np.random.randn(image_pixels, hidden_nodes)
    W2 = std2 * np.random.randn(hidden_nodes, class_size)
    b1 = std1 * np.random.randn(hidden_nodes)
    b2 = std2 * np.random.randn(class_size)
    return W1, W2, b1, b2


def sigmoid(x, clip):
    if clip:
        x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def softmax(x):
    # 1行（=1枚）ごとにソフトマックスをとる。元のプログラムの1枚分の計算と同じ
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def run(X, normalize, same_std, clip):
    # 指定した設定で1000枚を計算し、予測クラスと中間層の出力を返す
    W1, W2, b1, b2 = make_weights(same_std)
    x = X.reshape(len(X), -1).astype(np.float64)
    if normalize:
        x = x / pixel_value
    hidden = sigmoid(np.dot(x, W1) + b1, clip)
    output = softmax(np.dot(hidden, W2) + b2)
    return np.argmax(output, axis=1), hidden


def class_breakdown(pred):
    # 出力されたクラスを多い順に「クラス：回数」で並べる
    classes, counts = np.unique(pred, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return "、".join(f"{classes[i]}：{counts[i]}" for i in order)


# (表の行の名前, 正規化する, 標準偏差を層で共通にする, np.clipを使う)
SETTINGS = [
    ("前",                   False, True,  False),
    ("後",                   True,  False, True),
    ("後（正規化なし）",       False, False, True),
    ("後（標準偏差を層で共通）", True,  True,  True),
    ("後（np.clip なし）",     True,  False, False),
]


def check_with_original(images, preds):
    # 元のプログラムに実際に画像番号を入力し、上の計算と同じ答えになるかを確かめる
    for name, pred in preds.items():
        program = ROOT / "report" / "src" / f"kadai1_{name}.py"
        for n in images:
            result = subprocess.run([sys.executable, str(program)], input=f"{n}\n",
                                    capture_output=True, text=True, cwd=ROOT)
            printed = int(result.stdout.split()[-1]) #最後に出力される数字が認識結果
            mark = "一致" if printed == pred[n - 1] else "不一致"
            print(f"  {program.name} に {n:4d} を入力 → {printed}（まとめて計算：{pred[n - 1]}）{mark}")


def main():
    X_all = mnist.download_and_parse_mnist_file(str(DATA_DIR / "train-images-idx3-ubyte.gz"))
    Y_all = mnist.download_and_parse_mnist_file(str(DATA_DIR / "train-labels-idx1-ubyte.gz"))
    X = X_all[image_numbers] #元のプログラムは X[入力した番号] を使うので、そのまま番号で取り出す
    labels = Y_all[image_numbers]

    print(f"画像番号 {image_numbers[0]}〜{image_numbers[-1]}（{len(X)}枚）、シード値 {seed}")
    print(f"{'プログラム':<16}  正解率   飽和率   出力されたクラスの内訳")
    preds = {}
    with np.errstate(over="ignore"): #np.clip なしの版で出る警告を表示しない
        for name, normalize, same_std, clip in SETTINGS:
            pred, hidden = run(X, normalize, same_std, clip)
            accuracy = np.mean(pred == labels) * 100
            # 飽和率：中間層の出力のうち 0.01 未満または 0.99 より大きいものの割合
            saturation = np.mean((hidden < 0.01) | (hidden > 0.99)) * 100
            print(f"{name:<16}  {accuracy:5.1f}%  {saturation:5.1f}%   {class_breakdown(pred)}")
            preds[name] = pred

    print("\n元のプログラムを1枚ずつ実行した結果との照合（数枚のみ）")
    check_with_original([1, 100, 500, 1000], {"before": preds["前"], "after": preds["後"]})


if __name__ == "__main__":
    main()

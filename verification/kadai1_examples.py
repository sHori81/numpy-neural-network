# 課題1の実行例（レポート 2.4 節）を作るプログラム
#
# リファクタリング後のプログラム report/src/kadai1_after.py に画像番号を1つずつ入力して実際に実行し、
# 出力された認識結果と正解ラベルを表示する。
# また、入力した画像を report/fig/input_番号.png として保存する（レポートの図に使う）。
#
# 実行方法（リポジトリのフォルダで）：python verification/kadai1_examples.py

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg") #画面に表示せず、ファイルに保存する
import matplotlib.pyplot as plt
import mnist

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "mnist_data" / "train"
PROGRAM = ROOT / "report" / "src" / "kadai1_after.py"
FIG_DIR = ROOT / "report" / "fig"

image_numbers = [1, 4, 100, 108, 777] #入力する画像番号（5回分）


def main():
    X = mnist.download_and_parse_mnist_file(str(DATA_DIR / "train-images-idx3-ubyte.gz"))
    Y = mnist.download_and_parse_mnist_file(str(DATA_DIR / "train-labels-idx1-ubyte.gz"))
    FIG_DIR.mkdir(exist_ok=True)

    for n in image_numbers:
        result = subprocess.run([sys.executable, str(PROGRAM)], input=f"{n}\n",
                                capture_output=True, text=True, cwd=ROOT)
        output = result.stdout.split()[-1] #最後に出力される数字が認識結果
        print(f"画像番号 {n:4d}：出力 {output}、正解ラベル {Y[n]}")

        # 元のプログラムと同じく X[番号] の画像を保存する
        fig, ax = plt.subplots(figsize=(1.2, 1.2))
        ax.imshow(X[n], cmap="gray")
        ax.axis("off")
        fig.savefig(FIG_DIR / f"input_{n}.png", bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()

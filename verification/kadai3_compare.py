# 課題3のリファクタリング前後の比較（課題3レポートの表・図）を再現するプログラム
#
# report/src/kadai3_before.py（リファクタリング前）と report/src/kadai3_after.py（リファクタリング後）を
# そのまま読み込み、エポック数・学習率・正規化の有無だけを差し替えて main() を実行する。
# グラフの表示 (plot_history) は、学習の履歴を受け取るだけの関数に差し替える。
#
# 実験
#   A: リファクタリング前（正規化なし、学習率0.01）50エポック
#   B: リファクタリング後（正規化あり、学習率0.01）50エポック
#   C: リファクタリング後から正規化だけを外した版 3エポック → A の最初の3エポックと全桁一致するか確かめる
#   D: リファクタリング後（正規化あり、学習率0.1）50エポック
#
# 1回の学習（50エポック）に約10分かかるので、結果は verification/results/ に保存し、次回からはそれを使う。
# 実行方法（リポジトリのフォルダで）：
#   python verification/kadai3_compare.py          （保存済みの結果があれば再利用）
#   python verification/kadai3_compare.py --force  （全部計算し直す）

import contextlib
import io
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg") #ウィンドウを開かずに画像だけ作る
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "report" / "src"
RESULT_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = ROOT / "report" / "fig"
sys.path.insert(0, str(ROOT)) #mnist モジュールを読み込めるようにする

EXPERIMENTS = {
    #名前: (ファイル, エポック数, 学習率, 正規化するか, 説明)
    "A": ("kadai3_before.py", 50, 0.01, False, "前（正規化なし、学習率0.01）"),
    "B": ("kadai3_after.py", 50, 0.01, True, "後（正規化あり、学習率0.01）"),
    "C": ("kadai3_after.py", 3, 0.01, False, "後から正規化だけ外した版"),
    "D": ("kadai3_after.py", 50, 0.1, True, "後（正規化あり、学習率0.1）"),
}


def run(file_name, epochs, learning_rate, normalize):
    #元のプログラムを読み込み、設定を差し替えて学習し、履歴を返す
    source = (SRC_DIR / file_name).read_text(encoding="utf-8")
    if file_name == "kadai3_before.py" and normalize:
        raise ValueError("リファクタリング前は正規化なしでのみ実行する")

    namespace = {"__name__": "kadai3", "__file__": str(ROOT / "layers.py")} #MNIST をリポジトリの mnist_data から読む
    exec(compile(source, file_name, "exec"), namespace) #ここでシード値0が設定される

    namespace["epochs"] = epochs
    namespace["learning_rate"] = learning_rate
    if not normalize:
        namespace["pixel_value"] = 1.0 #リファクタリング後の preprocess は / pixel_value するので、1.0 で割る＝正規化なし

    captured = {}
    namespace["plot_history"] = lambda history, *args, **kwargs: captured.update(history)

    output = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(output):
        namespace["main"]()
    seconds = time.time() - start
    print(output.getvalue(), end="")
    return {key: np.array(value) for key, value in captured.items()}, seconds


def load_or_run(name, force):
    path = RESULT_DIR / f"kadai3_{name}.npz"
    if path.exists() and not force:
        data = np.load(path)
        return {key: data[key] for key in ("loss", "train_acc", "test_acc")}
    file_name, epochs, learning_rate, normalize, label = EXPERIMENTS[name]
    print(f"--- 実験{name}: {label}、{epochs}エポック")
    history, seconds = run(file_name, epochs, learning_rate, normalize)
    print(f"--- 実験{name}: {seconds:.0f}秒")
    RESULT_DIR.mkdir(exist_ok=True)
    np.savez(path, **history)
    return history


def plot_compare(results, names, file_name):
    #誤差と正答率の推移を、実験ごとに重ねて描く
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, color in zip(names, ["tab:blue", "tab:orange", "tab:green"]):
        history = results[name]
        epoch_axis = np.arange(1, len(history["loss"]) + 1)
        _, _, learning_rate, normalize, _ = EXPERIMENTS[name]
        label = f"{name}: {'before' if name == 'A' else 'after'}, {'normalized' if normalize else 'not normalized'}, lr={learning_rate}"
        ax_loss.plot(epoch_axis, history["loss"], color=color, label=label)
        ax_acc.plot(epoch_axis, history["test_acc"], color=color, label=f"{name} test")
        ax_acc.plot(epoch_axis, history["train_acc"], color=color, linestyle="--", label=f"{name} train")
    ax_loss.set_xlabel("Epochs"); ax_loss.set_ylabel("Cross entropy (epoch average)")
    ax_acc.set_xlabel("Epochs"); ax_acc.set_ylabel("Accuracy")
    ax_loss.legend(); ax_acc.legend(ncol=2)
    ax_loss.grid(alpha=0.3); ax_acc.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / file_name, dpi=150)
    plt.close(fig)


def plot_single(history, file_name):
    #1つの学習の誤差と正答率の推移を描く（元のプログラムの plot_history と同じ形）
    epoch_axis = np.arange(1, len(history["loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_loss.plot(epoch_axis, history["loss"], label="Loss")
    ax_loss.set_xlabel("Epochs"); ax_loss.set_ylabel("Cross entropy (epoch average)")
    ax_acc.plot(epoch_axis, history["train_acc"], label="Train Accuracy")
    ax_acc.plot(epoch_axis, history["test_acc"], label="Test Accuracy")
    ax_acc.set_xlabel("Epochs"); ax_acc.set_ylabel("Accuracy")
    for ax in (ax_loss, ax_acc):
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / file_name, dpi=150)
    plt.close(fig)


def main():
    force = "--force" in sys.argv
    results = {name: load_or_run(name, force) for name in EXPERIMENTS}

    print("\n=== 50エポック後の結果")
    print("実験  誤差     学習データ正答率  テストデータ正答率  テスト正答率の最大値(エポック)")
    for name in ("A", "B", "D"):
        h = results[name]
        best = int(np.argmax(h["test_acc"]))
        print(f"{name}     {h['loss'][-1]:.4f}   {h['train_acc'][-1]:.4f}            {h['test_acc'][-1]:.4f}"
              f"              {h['test_acc'][best]:.4f} ({best + 1})")

    print("\n=== 構造の変更で振る舞いが変わっていないか（A と C の最初の3エポック）")
    for epoch in range(3):
        a = [results["A"][key][epoch] for key in ("loss", "train_acc", "test_acc")]
        c = [results["C"][key][epoch] for key in ("loss", "train_acc", "test_acc")]
        print(f"epoch {epoch + 1}: A={a!r}  C={c!r}  一致={a == c}")

    print("\n=== 1エポック目の結果")
    for name in ("A", "B", "D"):
        h = results[name]
        print(f"{name}: loss {h['loss'][0]:.4f}  train {h['train_acc'][0]:.4f}  test {h['test_acc'][0]:.4f}")

    plot_single(results["B"], "kadai3_after.png")
    plot_compare(results, ["A", "B"], "kadai3_before_after.png")
    plot_compare(results, ["A", "B", "D"], "kadai3_learning_rate.png")
    print("\n図を保存しました：report/fig/kadai3_after.png, kadai3_before_after.png, kadai3_learning_rate.png")


if __name__ == "__main__":
    main()

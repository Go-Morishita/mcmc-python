import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math


def target_dist(x):
    return np.sin(2 * np.pi * x) + 1


def get_save_interval(i):
    if i <= 1000:
        return 100
    exponent = math.floor(math.log10(max(i, 1)))
    return 10 ** (exponent - 1)


samples = []
all_samples = []
num_steps = 1000000
x = 0.5  # 初期値
proposal_std = 0.1  # ジャンプ幅

for i in range(num_steps):
    # 今回は正規分布を提案分布とする
    x_proposal = np.random.normal(x, proposal_std)

    if not (0 <= x_proposal <= 1):
        continue

    acceptance_ratio = target_dist(x_proposal) / target_dist(x)
    if np.random.rand() < acceptance_ratio:
        x = x_proposal

    if i % 100 == 0:
        samples.append(x)

    if i % get_save_interval(i) == 0:
        all_samples.append(samples.copy())

# 描画ようのFigureと軸(ax)を作成
fig, ax = plt.subplots()

# f(x)の描画
xs = np.linspace(0, 1, 500)
ys = target_dist(xs)
ys /= np.trapezoid(ys, xs)
line, = ax.plot(xs, ys, color='red', label='Target Distribution')

# 初期ヒストグラムを1回描画（凡例のため）
# ax.histは3成分の戻り値がある, 第3成分が実際の棒
hist = ax.hist(all_samples[0], bins=50, density=True,
               alpha=0.6, label='Samples')[2]

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 2.5)
ax.set_title("MCMC Sampling Progress")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()

# フレーム更新関数


def update(frame):
    global hist
    if hist:
        for bar in hist:
            bar.remove()
    hist = ax.hist(all_samples[frame], bins=40,
                   density=True, alpha=0.6, color='C0')[2]
    ax.set_title(f"Step: {frame * 100}")


# アニメーション作成
ani = FuncAnimation(fig, update, frames=len(
    all_samples), repeat=False, interval=50)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 目標分布, 2次元のガウス分布
def target_dist(x, y):
    return np.exp(-0.5 * (x**2 + y**2 + 0.8 * x * y))

# サンプリングの初期設定
num_samples = 5000
samples = np.zeros((num_samples, 2))
x, y = 0.0, 0.0
proposal_std = 0.5 # 提案分布の標準偏差, ジャンプ幅を制御

# Hasting and Gibbs Sampling
for i in range(num_samples):
    # xの更新
    x_prop = np.random.normal(x, proposal_std)
    if np.random.rand() < target_dist(x_prop, y) / target_dist(x, y):
        x = x_prop
        
    # yの更新
    y_prop = np.random.normal(y, proposal_std)
    if np.random.rand() < target_dist(x, y_prop) / target_dist(x, y):
        y = y_prop
        
    samples[i] = [x, y]

# グリッドの生成
x_vals = np.linspace(-3, 3, 200)
y_vals = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x_vals, y_vals)

# 目標分布の評価
Z = target_dist(X, Y)

# 周辺化のための数値積分を行う
dx = x_vals[1] - x_vals[0]
dy = y_vals[1] - y_vals[0]
marginal_x = Z.sum(axis=0) * dy
marginal_y = Z.sum(axis=1) * dx

# 周辺密度を確率として正規化
area_x = np.trapezoid(marginal_x, x_vals)
area_y = np.trapezoid(marginal_y, y_vals)
marginal_x_norm = marginal_x / area_x
marginal_y_norm = marginal_y / area_y

# スケーリングのために最大値を格納
max_marginal_x = np.max(marginal_x_norm)
max_marginal_y = np.max(marginal_y_norm)

# 3Dプロットの設定
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 軸の範囲を設定
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
max_pdf = Z.max()
ax.set_zlim(0, max_pdf)

# z=0の平面を描画
Z0 = np.zeros_like(Z)
ax.plot_surface(
    X, Y, Z0,
    alpha=0.0,
    rstride=20, cstride=20,
    edgecolor='none'
)

# サンプルを描画
ax.scatter(
    samples[:, 0], samples[:, 1], np.zeros(num_samples),
    color='red', s=5, alpha=0.4, label='Samples'
)

# 壁の生成
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

# marginal_x_norm は積分すると1になるように正規化されているが,
# ピークが1を超えている可能性があるので max_marginal_xを使って再度正規化を行う.
# その後, 目標分布の最大値で拡大する.
wall_y = np.full_like(x_vals, y_max)
ax.plot(
    x_vals, wall_y, marginal_x_norm / max_marginal_x * max_pdf,
    color='blue', lw=2, label='p(x) (normalized)'
)

# ここも上と同様に正規化を行う.
wall_x = np.full_like(y_vals, x_min)
ax.plot(
    wall_x, y_vals, marginal_y_norm / max_marginal_y * max_pdf,
    color='green', lw=2, label='p(y) (normalized)'
)

# 軸のラベルとタイトルを設定
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Density")
ax.set_title("Hesting and Gibbs Sampling")
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

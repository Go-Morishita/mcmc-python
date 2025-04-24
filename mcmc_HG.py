import numpy as np
import matplotlib.pyplot as plt

# 目標分布
def target_dist(x, y):
    return np.exp(-0.5 * (x**2 + y**2 + 0.8 * x * y))

# --- MCMC サンプリング ---
num_samples = 5000
samples = np.zeros((num_samples, 2))
x, y = 0.0, 0.0
proposal_std = 0.5

for i in range(num_samples):
    # x の更新
    x_prop = np.random.normal(x, proposal_std)
    if np.random.rand() < target_dist(x_prop, y) / target_dist(x, y):
        x = x_prop
    # y の更新
    y_prop = np.random.normal(y, proposal_std)
    if np.random.rand() < target_dist(x, y_prop) / target_dist(x, y):
        y = y_prop
    samples[i] = [x, y]

# --- メッシュと PDF サーフェスの生成 ---
x_vals = np.linspace(-3, 3, 200)
y_vals = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = target_dist(X, Y)

# --- 周辺分布の計算 ---
dx = x_vals[1] - x_vals[0]
dy = y_vals[1] - y_vals[0]
marginal_x = Z.sum(axis=0) * dy      # ∫ p(x,y) dy
marginal_y = Z.sum(axis=1) * dx      # ∫ p(x,y) dx

# 正規化
area_x = np.trapezoid(marginal_x, x_vals)
area_y = np.trapezoid(marginal_y, y_vals)
marginal_x_norm = marginal_x / area_x
marginal_y_norm = marginal_y / area_y

# 最大値を取得
max_marginal_x = np.max(marginal_x_norm)
max_marginal_y = np.max(marginal_y_norm)

# --- プロット ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 軸範囲をメッシュの定義域に合わせる
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())

# z方向は PDF の最大値にぴったり合わせる
max_pdf = Z.max()
ax.set_zlim(0, max_pdf)

# 1) 底面：Z=0 の面を、X,Y のメッシュそのままに
Z0 = np.zeros_like(Z)
ax.plot_surface(
    X, Y, Z0,
    alpha=0.0,
    rstride=20, cstride=20,
    edgecolor='none'
)

# 3) サンプル点 (z=0 上にプロット)
ax.scatter(
    samples[:, 0], samples[:, 1], np.zeros(num_samples),
    color='red', s=5, alpha=0.4, label='Samples'
)

# 壁面位置の取得
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

# 4) p(x) を y = 奥の壁 (y_max) 面に
wall_y = np.full_like(x_vals, y_max)
ax.plot(
    x_vals, wall_y, marginal_x_norm / max_marginal_x * max_pdf,
    color='blue', lw=2, label='p(x) (normalized)'
)

# 5) p(y) を x = 手前の壁 (x_min) 面に
wall_x = np.full_like(y_vals, x_min)
ax.plot(
    wall_x, y_vals, marginal_y_norm / max_marginal_y * max_pdf,
    color='green', lw=2, label='p(y) (normalized)'
)

# ラベル等
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Density")
ax.set_title("Hesting and Gibbs Sampling")
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

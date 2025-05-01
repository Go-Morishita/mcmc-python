# こっちはできてる

import numpy as np
import matplotlib.pyplot as plt

# --- モデルパラメータ -------------------------------------------------
a, b = 25.0, 0.05          # 周波数と位相
R_MAX = np.sqrt(2)/2       # 単位正方形＋周期境界で取り得る最大距離 ≈0.707

# --- κ2(r) の定義 -----------------------------------------------------
def S_const(a, b):
    num = (a**2 + 1)**2 * (b**2 + 2*b + 2)
    den = 2*((a**2 + 1)**2*(b + 1) - (a**2 + 1)*(b - 1) - 2)
    return num / den

def Cab(a, b):
    S = S_const(a, b)
    term1 = (b - 1)/((a**2 + 1)*(b + 1))
    term2 = 2/((a**2 + 1)**2*(b + 1))
    term3 = b**2/(2*(b + 1)*S)
    return term1 + term2 + term3

C_ab = Cab(a, b)

def kappa2(r):
    """論文式(6.1) に対応する二次相関 κ₂(r)"""
    if r < b:
        return -1.0
    return np.exp(-(r - b)) * (-np.cos(a*(r - b)) + C_ab)

# --- 目標密度 (正規化前) ---------------------------------------------
def target_density(r):
    val = 1.0 + kappa2(r)   # n=2 なので 1 + κ₂(r)
    return max(val, 0.0)    # 漏れ防止（理論上 0 以上）

# --- MH サンプリング --------------------------------------------------
num_steps     = 1_000_000_0
proposal_std  = 0.05
current_r     = 0.35        # 初期値は b～R_MAX の間で適当に
samples = []

for i in range(num_steps):
    r_prop = np.random.normal(current_r, proposal_std)

    # 定義域チェック（0 < r < R_MAX）
    if not (0.0 < r_prop < R_MAX):
        continue

    acc_ratio = target_density(r_prop) / target_density(current_r)
    if np.random.rand() < acc_ratio:
        current_r = r_prop

    if i % 100 == 0:        # 100 ステップごとに保存
        samples.append(current_r)

# --- 可視化 -----------------------------------------------------------
xs = np.linspace(0, R_MAX, 1000)
ys = np.vectorize(target_density)(xs)
ys /= np.trapezoid(ys, xs)          # 数値積分で正規化

plt.hist(samples, bins=120, density=True, alpha=0.6, label='Samples')
plt.plot(xs, ys, color='red', label='Target')
plt.xlabel('粒子間距離 r')
plt.ylabel('確率密度')
plt.title('加法型モデル + MH (a=25, b=0.05)')
plt.legend()
plt.show()

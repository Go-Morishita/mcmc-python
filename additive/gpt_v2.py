import numpy as np
import matplotlib.pyplot as plt

# --- モデルパラメータ ---
OMEGA = 1.0               # トーラスの一辺長
REJECT_POINT = 0.05       # kappa2 が -1 を返す閾値
a = 25.0
b = 0.05

# --- 事前計算関数 ---
def S_function():
    return ((a**2 + 1)**2 * (b**2 + 2*b + 2)) / (
        2 * ((a**2 + 1)**2 * (b + 1) - (a**2 + 1)*(b - 1) - 2)
    )

def C_function():
    return (b - 1)/( (a**2 +1)*(b+1) ) + 2/((a**2 +1)**2*(b+1)) + b**2/(2*(b+1)*S_function())

# --- 二点間相互作用カーネル ---
def kappa2(r):
    mask = r < REJECT_POINT
    val = np.exp(-(r - b)) * (-np.cos(a*(r - b)) + C_function())
    return np.where(mask, -1.0, val)

# --- 抑制項（n=2のときは簡易化されます） ---
def suppression_term(r):
    # n=2 のとき、組み合わせ数 n(n-1)/2 = 1 なので
    return - kappa2(r) / ( S_function() * (C_function() - 1) )

# --- n=2 用の p_additive ---
def p_2particle(pos):
    # pos: shape (2,2)  → [[x1,y1],[x2,y2]]
    # １次の一様項
    uniform_term = 1.0 / OMEGA**2

    # ２次の相互作用項
    # 距離は周期境界で測る
    delta = np.abs(pos[0] - pos[1])
    delta = np.minimum(delta, OMEGA - delta)
    r = np.hypot(*delta)

    interaction = kappa2(r)
    suppress   = suppression_term(r)

    return uniform_term + interaction + suppress

# --- トーラス境界の距離計算 ---
def toroidal_distance(x1, x2):
    d = np.abs(x2 - x1)
    d = np.minimum(d, OMEGA - d)
    return np.hypot(*d)

# --- Metropolis–Hastings for 2 particles ---
def metropolis_hastings_2particles(num_steps, proposal_std, sample_interval=100):
    # 初期化：ランダムにトーラス上に２点を配置
    positions = np.random.rand(2,2) * OMEGA
    current_p = p_2particle(positions)

    samples = []
    for step in range(num_steps):
        # 提案：両粒子それぞれにガウス摂動
        proposal = np.random.normal(positions, scale=proposal_std, size=positions.shape)
        proposal %= OMEGA  # 周期境界条件

        prop_p = p_2particle(proposal)
        alpha  = min(1.0, prop_p / (current_p + 1e-16))

        if np.random.rand() < alpha:
            positions, current_p = proposal, prop_p

        if step % sample_interval == 0:
            samples.append(positions.copy())

    return np.array(samples)  # shape=(num_samples, 2, 2)

# --- サンプリング実行 ---
num_steps       = 200_000
proposal_std    = 0.05
sample_interval = 200

samples = metropolis_hastings_2particles(num_steps, proposal_std, sample_interval)

# --- 距離の抽出とヒストグラム ---
# 各サンプルから距離 r を計算
distances = np.array([
    toroidal_distance(s[0], s[1])
    for s in samples
])

# ヒストグラム
bins = np.linspace(0, np.sqrt(2)*OMEGA/2, 50)
hist, edges = np.histogram(distances, bins=bins, density=True)
centers = 0.5*(edges[:-1] + edges[1:])

plt.figure(figsize=(6,4))
plt.bar(centers, hist, width=edges[1]-edges[0], alpha=0.6, label='Empirical g(r)')

# 理論的な kappa2(r) を同プロット
r_plot = np.linspace(0, centers.max(), 500)
plt.plot(r_plot, kappa2(r_plot) + suppression_term(r_plot), 'r-', label='Interaction kernel')

plt.xlabel('Distance $r$')
plt.ylabel('Density')
plt.title('2-Particle Correlation Function (MH Sampling)')
plt.legend()
plt.tight_layout()
plt.show()

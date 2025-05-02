# 次回：確率密度関数を定義して実際にMH法とHWG法でサンプリングしてみる.

# C:\Users\morishita\CGPrograming\MCMC\AGGeneratingParticlesWithCorrelations\taichi_calculator_ex.py
# これをみるのがいい, まずランダムな位置を生成して、次にそれをMH法でサンプリングしている.
# C:\Users\morishita\CGPrograming\MCMC\AGGeneratingParticlesWithCorrelations\taichi_call_st_ex.pyこれも実装したほうがいいかも.

# 2024年5月1日
# 結構いい感じできたが, 細かい部分がテキトウ
# 若干2つ目の山が低い気がする, 少し確認すべき
# 次のステップはtichiを導入するか, MWGでやる
# ☆chainsを理解して実装すべき. 多分ここがtaichiと繋がてくる, 先輩のコードを見るべき

# taichiはできたので, burn-inを実装する

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import taichi as ti

# -------------------------- 定数 --------------------------
OMEGA = 1.0
SIDE = np.sqrt(OMEGA)
REJECT_POINT = 0.05
PROPOSAL_STD = 0.05
SCALING_FACTOR = 6.0
GETA = 6.0
a = 25.0
b = 0.05
NUM_PARTICLES = 2
num_chains = 1000 # 並列化のためのチェーン数
num_steps = 10000
STORE_INTERVAL = 100 # サンプリング間隔

K2_SAMPLES = 1000

# -------------------------- Taichi field --------------------------
# Taichiの初期化
ti.init(arch=ti.cuda, random_seed=42) # GPUを使用する場合はti.cudaを指定
# 粒子位置
particles_positions = ti.Vector.field(2, dtype=ti.f32, shape=(num_chains, NUM_PARTICLES))
# 提案位置
proposal_positions = ti.Vector.field(2, dtype=ti.f32, shape=(num_chains, NUM_PARTICLES))
# 現在の確率
current_probability = ti.field(dtype=ti.f32, shape=(num_chains))
# 保存用
num_records = num_steps // STORE_INTERVAL
samples = ti.field(dtype=ti.f32, shape=(num_chains, num_records))
# Kappa2の計算用
kappa2_values = ti.Vector.field(2, dtype=ti.f32, shape=(K2_SAMPLES))

@ti.func
def S_function():
    return ((a** 2 + 1)** 2 * (b** 2 + 2 * b + 2)) / (2 * ((a** 2 + 1)** 2 * (b + 1) - (a** 2 + 1) * (b - 1) - 2))

@ti.func
def C_function():
    return (b - 1) / ((a** 2 + 1) * (b + 1)) + 2 / ((a** 2 + 1)** 2 * (b + 1)) + b** 2 / (2 * (b + 1) * S_function())

@ti.func
def kappa2(r):
    return ti.exp(- (r - b)) * (- ti.cos(a * (r - b)) + C_function())

@ti.func
def cal_kappa2(r):
    return ti.select(r < REJECT_POINT, -1.0, SCALING_FACTOR * kappa2(r) + GETA) # REJECT_POINT以下は0にする
    
@ti.func
def cal_sigma(positions, chain: ti.i32): # type: ignore[valid-type]
    sigma = 0.0
    for i in range(NUM_PARTICLES - 1):
        for j in range(i + 1, NUM_PARTICLES):
            # 境界周期条件を考慮して、位置を更新する
            dr = ti.abs((positions[chain, i] - positions[chain, j]))
            dr = ti.min(dr, OMEGA - dr)
            sigma += cal_kappa2(dr.norm())
    return sigma

@ti.func
def suppression_term(positions, chain: ti.i32): # type: ignore[valid-type]       
    return - cal_sigma(positions, chain) / (S_function() * NUM_PARTICLES * (NUM_PARTICLES - 1) * 0.5 * (C_function() - 1))

@ti.func
def p_additive(positions, chain: ti.i32): # type: ignore[valid-type]
    return 1 / ti.abs(OMEGA)** NUM_PARTICLES + 1 / ti.abs(OMEGA)** (NUM_PARTICLES - 2) * cal_sigma(positions, chain) + suppression_term(positions, chain)

@ti.kernel
def metrpolis_hastings_onestep(step_idx: ti.i32): # type: ignore[valid-type]
    for chain in range(num_chains):
        # 提案位置の生成
        for particle in ti.static(range(NUM_PARTICLES)): # ti.staticはコンパイラに対して、ループの長さが固定であることを伝える
            delta = ti.randn() * PROPOSAL_STD
            position = particles_positions[chain, particle] + ti.Vector([delta, ti.randn()*PROPOSAL_STD])
            proposal_positions[chain, particle] = (position + SIDE) % SIDE  # 周期境界条件を考慮して位置を更新する
        
        proposal_probability = p_additive(proposal_positions, chain)
        
        # 受容判定
        if ti.random() < proposal_probability / current_probability[chain]:
            for particle in ti.static(range(NUM_PARTICLES)):
                particles_positions[chain, particle] = proposal_positions[chain, particle]
            current_probability[chain] = proposal_probability
        
        # レコード保存    
        if step_idx % STORE_INTERVAL == 0:
            recprded_idx = step_idx // STORE_INTERVAL
            dist = ti.abs((particles_positions[chain, 0] - particles_positions[chain, 1]))
            dist = ti.min(dist, SIDE - dist) # 周期境界条件を考慮して距離を更新する
            samples[chain, recprded_idx] = dist.norm() # 2次元ベクトルのノルムを計算する

@ti.kernel
def init():
    for x, y in particles_positions:
        particles_positions[x, y] = ti.random() * SIDE

@ti.kernel
def init_k2_curve():
    for i in range(K2_SAMPLES):
        r = 0.7 * i / (K2_SAMPLES - 1)       # 0 〜 0.7 の等間隔
        kappa2_values[i] = ti.Vector([r, cal_kappa2(ti.cast(r, ti.f32))])
        
# -------------------------- 初期化 --------------------------
init()

# --------------------------Metropolis-Hastings法の実行--------------------------
for step_idx in tqdm(range(num_steps)):
    metrpolis_hastings_onestep(step_idx)
    
samples_np = samples.to_numpy().reshape(-1)

# --------------------------サンプリング結果のプロット--------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --------------------------正規化なし--------------------------
axes[0].hist(samples_np, bins=50, density=False, alpha=0.6, label='Samples')
axes[0].set_xlim(0, 0.7)
axes[0].legend()
axes[0].set_title("Distanses between two particles")
axes[0].set_xlabel("Distance")
axes[0].set_ylabel("Density")

# --------------------------正規化あり--------------------------
init_k2_curve()
kappa2_np = kappa2_values.to_numpy()
xs = kappa2_np[:, 0]
ys = np.maximum(kappa2_np[:, 1], 0.0)  # 負の値を0に設定

# samplesの正規化
hist, bin_edges = np.histogram(samples_np, bins=200, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

nomalized_samples = hist / bin_centers # 円周効果を取り除き, 真の密度を求める

axes[1].bar(bin_centers, nomalized_samples, width=bin_edges[1]-bin_edges[0], alpha=0.6, label='Empirical Density')
axes[1].plot(xs, ys, label='K$_2$', color='red')
axes[1].set_xlim(0, 0.7)
axes[1].legend()
axes[1].set_title("Normalized Distanses between two particles")
axes[1].set_xlabel("Distance")
axes[1].set_ylabel("Normalized Density")

plt.tight_layout()
plt.show()
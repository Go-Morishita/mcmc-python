# 次回：確率密度関数を定義して実際にMH法とHWG法でサンプリングしてみる.

# C:\Users\morishita\CGPrograming\MCMC\AGGeneratingParticlesWithCorrelations\taichi_calculator_ex.py
# これをみるのがいい, まずランダムな位置を生成して、次にそれをMH法でサンプリングしている.
# C:\Users\morishita\CGPrograming\MCMC\AGGeneratingParticlesWithCorrelations\taichi_call_st_ex.pyこれも実装したほうがいいかも.

# 2024年5月1日
# 結構いい感じできたが, 細かい部分がテキトウ
# 若干2つ目の山が低い気がする, 少し確認すべき
# 次のステップはtichiを導入するか, MWGでやる
# ☆chainsを理解して実装すべき. 多分ここがtaichiと繋がてくる, 先輩のコードを見るべき

import numpy as np
import matplotlib.pyplot as plt

def S_function():
    return ((a** 2 + 1)** 2 * (b** 2 + 2 * b + 2)) / (2 * ((a** 2 + 1)** 2 * (b + 1) - (a** 2 + 1) * (b - 1) - 2))

def C_function():
    return (b - 1) / ((a** 2 + 1) * (b + 1)) + 2 / ((a** 2 + 1)** 2 * (b + 1)) + b** 2 / (2 * (b + 1) * S_function())

def kappa2(r):
    return np.exp(- (r - b)) * (- np.cos(a * (r - b)) + C_function())

def cal_kappa2(r, scaling_factor, geta):
    if r < REJECT_POINT:
        return -1.0
    else:
        return scaling_factor * kappa2(r) + geta

def suppression_term(positions, scaling_factor, geta):
    n = len(positions)
    
    sigma = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dr = np.abs(positions[i] - positions[j])
            dr = np.minimum(dr, OMEGA - dr)
            r = np.hypot(*dr)
            sigma += cal_kappa2(r, scaling_factor, geta)
            
    return - sigma / (S_function() * n * (n - 1) * 0.5 * (C_function() - 1))

def p_additive(positions, scaling_factor, geta):
    n = len(positions)
    
    sigma = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            # 境界周期条件を考慮して、位置を更新する
            dr = np.abs(positions[i] - positions[j])
            dr = np.minimum(dr, OMEGA - dr)
            r = np.hypot(*dr) # 距離を返す：√(x**2 + y**2)
            sigma += cal_kappa2(r, scaling_factor, geta)
    
    return 1 / np.abs(OMEGA)** n + 1 / np.abs(OMEGA)** (n - 2) * sigma + suppression_term(positions, scaling_factor, geta)

def metrpolis_hastings(num_particles, num_steps, proposal_std, scaling_factor, geta):
    # 初期化
    side = np.sqrt(OMEGA)
    positions = np.random.rand(num_particles, 2) * side
    samples = []
    current_p = p_additive(positions, scaling_factor, geta)
    
    for step in range(num_steps):
        proposals = positions + np.random.normal(scale=proposal_std, size=positions.shape)
        proposals = proposals % side
        
        proposal_p = p_additive(proposals, scaling_factor, geta)
        
        acceptance_ratio = proposal_p / current_p
        
        if np.random.rand() < acceptance_ratio:
            positions, current_p = proposals, proposal_p
        
        if step % 100 == 0:
            r = np.linalg.norm(positions[0] - positions[1])
            samples.append(r)
        
    return np.array(samples)

if __name__ == "__main__":
    
    # --------------------------乱数のシードを設定--------------------------
    np.random.seed(42)
    
    # --------------------------定数の設定--------------------------
    OMEGA = 1.0
    REJECT_POINT = 0.05
    a = 25.0
    b = 0.05
    num_particles = 2
    num_steps = 1000000
    proposal_std = 0.05
    scaling_factor = 5.0
    geta = 5.0
    
    # --------------------------Metropolis-Hastings法の実行--------------------------
    samples = metrpolis_hastings(num_particles, num_steps, proposal_std, scaling_factor, geta)
    
    # --------------------------サンプリング結果のプロット--------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --------------------------正規化なし--------------------------
    axes[0].hist(samples, bins=50, density=False, alpha=0.6, label='Samples')
    axes[0].set_xlim(0, 0.7)
    axes[0].legend()
    axes[0].set_title("Distanses between two particles")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Density")
    
    # --------------------------正規化あり--------------------------
    target_function_vectorized = np.vectorize(lambda r: max(cal_kappa2(r, scaling_factor, geta), 0), otypes=[float]) # 無理やり REJECT_POINT 以下を0にしているので改善の余地あり
    xs = np.linspace(0, 0.7, 1000)
    ys = target_function_vectorized(xs)
    
    # samplesの正規化
    hist, bin_edges = np.histogram(samples, bins=200, density=True)
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
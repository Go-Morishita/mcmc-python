import numpy as np
import matplotlib.pyplot as plt

def target_dist(x):
    return np.sin(2 * np.pi * x) + 1

samples = []
num_steps = 1000000
x = 0.5 # 初期値
proposal_std = 0.1 # ジャンプ幅

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

 # binsで棒の数, density=TRUEで確率密度そして正規化
plt.hist(samples, bins=50, density=True, alpha=0.6, label='Samples')
xs = np.linspace(0, 1, 500)
ys = target_dist(xs)
ys /= np.trapezoid(ys, xs)
plt.plot(xs, ys, label='Target Distribution', color='red')
plt.legend()
plt.title("MCMC Sampling of f(x) = sin(2πx) + 1")
plt.xlabel("x")
plt.ylabel("DEnsity")
plt.show()
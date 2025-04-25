# 次回：確率密度関数を定義して実際にMH法とHWG法でサンプリングしてみる.

import numpy as np
import matplotlib.pyplot as plt

REJECT_POINT = 0.05

def S_function(a, b):
    return ((a** 2 + 1)** 2 * (b** 2 + 2 * b + 2)) / (2 * ((a** 2 + 1)** 2 * (b + 1) - (a** 2 + 1) * (b - 1) - 2))

def C_function(a, b):
    return (b - 1) / ((a** 2 + 1) * (b + 1)) + 2 / ((a** 2 + 1)** 2 * (b + 1)) + b** 2 / (2 * (b + 1) * S_function(a, b))

def kappa2(r, a=25.00, b=0.05):
    if r < REJECT_POINT:
        return -1
    else:
        return np.exp(- (r - b)) * (- np.cos(a * (r - b)) + C_function(a, b))

def proposal_function():
    return 0

target_function_vectorized = np.vectorize(kappa2, otypes=[float])

xs = np.linspace(0, 0.7, 1000)
ys = target_function_vectorized(xs)
plt.plot(xs, ys, label='K$_2$', color='red')
plt.xlim(0, 0.7)
plt.legend()
plt.title("Target Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.show()
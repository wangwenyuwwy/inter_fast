import numpy as np
from scipy.stats import expon, norm

# 蒙特卡洛模拟（假设 X1~Exponential(1), X2~Normal(0,1), ...）
N = 1000000
samples = np.array([
    expon.rvs(scale=1, size=N),   # X1 ~ Exp(1)
    norm.rvs(loc=0, scale=1, size=N),  # X2 ~ N(0,1)
    norm.rvs(loc=0.5, scale=0.8, size=N),  # X2 ~ N(0,1)
    # ... 生成其他分布的样本
])

min_indices = np.argmin(samples, axis=0)
probabilities = np.bincount(min_indices, minlength=6) / N
print("概率估计:", probabilities)
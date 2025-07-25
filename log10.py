# Cell 1: 安装依赖并导入库
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
import pandas as pd

# Cell 2: 定义欧拉熵 H_K
def HK(K, t_max=200):
    integrand = lambda t: K * t**(K-1) * np.exp(-t) * np.log(K * t**(K-1) * np.exp(-t) + 1e-20)
    val, _ = quad(integrand, 0, t_max, limit=200)
    return -val

# Cell 3: 定义黎曼谱熵 H_sigma
def Hσ(σ, γn=None):
    if γn is None:
        γn = np.array([
            14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
            37.5862, 40.9187, 43.3271, 48.0052, 49.7738,
            52.9703, 56.4462, 59.3470, 60.8318, 65.1125,
            67.0798, 69.5464, 72.0672, 75.7047, 77.1448,
            79.3373, 82.9104, 84.7355, 87.4253, 88.8091,
            92.4919, 94.6513, 95.8706, 98.8312, 101.317
        ])
    fσ = np.exp(-γn**2 / (2 * σ**2))
    fσ /= np.sum(fσ)
    return -np.sum(fσ * np.log(fσ + 1e-20))

# Cell 4: 定义损失函数
def loss(params, loga_target):
    K, σ = params
    if K <= 0 or σ <= 0: return np.inf
    try:
        hk = HK(K)
        hσ = Hσ(σ)
        return (np.log(abs(hσ / hk)) - loga_target)**2
    except:
        return np.inf

# Cell 5: 设定常数 log(a)
const_log_values = {
    "G (log10)": -38.437,
    "α (log10)": -4.9255,
    "c (log10)": 19.5193,
    "h (log10)": -37.7308,
    "e": 1.0,
    "π": np.log(np.pi),
    "1/π": -np.log(np.pi),
    "√2": np.log(np.sqrt(2)),
    "φ": np.log((1 + np.sqrt(5)) / 2)
}

# Cell 6: 优化匹配结果
results = {}

for name, loga in const_log_values.items():
    res = minimize(lambda x: loss(x, loga), x0=[5.0, 10.0], method='Nelder-Mead')
    if res.success:
        K_opt, σ_opt = res.x
        hk = HK(K_opt)
        hσ = Hσ(σ_opt)
        log_H_ratio = np.log(abs(hσ / hk))
        delta = abs(log_H_ratio - loga)
        results[name] = {
            'log(a)': loga,
            'K': round(K_opt, 4),
            'σ': round(σ_opt, 4),
            'log(Hσ/HK)': round(log_H_ratio, 6),
            'Δ': round(delta, 6)
        }

# 输出结果表格
df = pd.DataFrame(results).T
display(df)

# Cell 7: 可视化
fig, ax = plt.subplots()
for name in results:
    plt.scatter(results[name]['σ'], results[name]['K'], label=name)
plt.xlabel('σ')
plt.ylabel('K')
plt.title('Optimized (σ, K) for log(Hσ / HK) ≈ log a')
plt.legend()
plt.grid(True)
plt.show()

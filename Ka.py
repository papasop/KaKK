import numpy as np
import pandas as pd
from scipy.integrate import quad
from tqdm.notebook import tqdm

# 原教旨谱熵函数 H_sigma(K, σ)
def H_sigma(K, sigma, t_max=200):
    integrand = lambda t: K * t**(K - 1) * np.exp(-t) * np.log(1 + sigma * t)
    result, _ = quad(integrand, 0, t_max, limit=500)
    return -result

# 原教旨欧拉熵函数 H_K(K)
def H_K(K, t_max=200):
    integrand = lambda t: K * t**(K - 1) * np.exp(-t) * np.log(K * t**(K - 1) * np.exp(-t) + 1e-20)
    result, _ = quad(integrand, 0, t_max, limit=500)
    return -result

# 常数轨道 log(a)
a_list = [1/2, 1/np.e, 1/np.pi, 1/1.61803,  # 压缩轨道
          np.sqrt(2), 2, np.e, 3, np.pi, 1.0]  # 扩张轨道 + C₁
labels = ["1/2", "1/e", "1/π", "1/φ", "√2", "2", "e", "3", "π", "𝓒₁"]
target_logs = np.log(a_list)

# 参数扫描范围
K_vals = np.linspace(3.5, 5.0, 80)       # K 扫描范围
sigma_vals = np.linspace(4.0, 10.0, 80)  # σ 扫描范围

# 结果表
results = []

for label, loga in tqdm(zip(labels, target_logs), total=len(labels)):
    min_error = np.inf
    best_K, best_sigma = None, None
    best_log_ratio = None

    for K in K_vals:
        HK = H_K(K)
        for sigma in sigma_vals:
            Hσ = H_sigma(K, sigma)
            if HK == 0 or Hσ == 0:
                continue
            log_ratio = np.log(np.abs(Hσ / HK))
            error = np.abs(log_ratio - loga)
            if error < min_error:
                min_error = error
                best_K = K
                best_sigma = sigma
                best_log_ratio = log_ratio

    if best_K is not None:
        results.append([label,
                        round(loga, 6),
                        round(best_K, 4),
                        round(best_sigma, 4),
                        round(best_log_ratio, 6),
                        round(min_error, 6)])
    else:
        results.append([label, round(loga, 6), None, None, None, None])

# 构造 DataFrame
df = pd.DataFrame(results, columns=["轨道", "log(a)", "K", "σ", "log(Hσ/HK)", "Δ"])
df
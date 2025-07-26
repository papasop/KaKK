import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from tqdm import tqdm

# ---------- Step 1: 定义结构熵函数 ----------
def H_sigma(K, sigma, t_max=200):
    integrand = lambda t: K * t**(K - 1) * np.exp(-t) * np.log(1 + sigma * t)
    result, _ = quad(integrand, 0, t_max, limit=500)
    return -result

def H_K(K, t_max=200):
    integrand = lambda t: K * t**(K - 1) * np.exp(-t) * np.log(K * t**(K - 1) * np.exp(-t) + 1e-20)
    result, _ = quad(integrand, 0, t_max, limit=500)
    return -result

# ---------- Step 2: 设置扫描参数 ----------
sigma_fixed = 8.7388
K_vals = np.round(np.arange(4.0931, 4.0935 + 0.00005, 0.00005), 5)
target_log = np.log(np.e)

# ---------- Step 3: 开始扫描 ----------
results = []
for K in tqdm(K_vals, desc="扫描 K 值"):
    HK = H_K(K)
    Hσ = H_sigma(K, sigma_fixed)
    log_ratio = np.log(np.abs(Hσ / HK))
    delta = np.abs(log_ratio - target_log)
    results.append((K, sigma_fixed, log_ratio, delta))

# ---------- Step 4: 整理并输出前10精度结果 ----------
df = pd.DataFrame(results, columns=["K", "σ", "log(Hσ/HK)", "Δ"])
df_sorted = df.sort_values(by="Δ").reset_index(drop=True)

print("Top 10 精度结果:")
print(df_sorted.head(10).to_string(index=False))

# ---------- Step 5: 可视化 ----------
plt.figure(figsize=(8, 4))
plt.plot(df["K"], df["Δ"], marker='o')
plt.title("Δ vs K（固定 σ = 8.7388）")
plt.xlabel("K")
plt.ylabel("Δ = |log(Hσ/HK) - log(e)|")
plt.grid(True)
plt.show()

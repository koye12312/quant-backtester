import numpy as np
import matplotlib.pyplot as plt

def gbm_paths(S0=100.0, mu=0.05, sigma=0.2, T=1.0, steps=252, n_paths=1000, seed=42):
    np.random.seed(seed)
    dt = T/steps
    Z = np.random.normal(size=(steps, n_paths))
    paths = np.empty((steps+1, n_paths))
    paths[0] = S0
    for t in range(1, steps+1):
        paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[t-1])
    return paths

def est_prob_final_ge_k(paths, K=110.0):
    final = paths[-1]
    return float((final >= K).mean())

if __name__ == "__main__":
    paths = gbm_paths()
    plt.figure()
    plt.plot(paths[:, :20])
    plt.title("Sample GBM Paths (first 20)")
    plt.xlabel("Step"); plt.ylabel("Price")
    plt.tight_layout()
    plt.show()
    print("P(S_T >= 110):", round(est_prob_final_ge_k(paths, 110.0), 4))
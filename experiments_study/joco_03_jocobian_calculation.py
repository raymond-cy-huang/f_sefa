import numpy as np


def phi(w):
    w1, w2 = w
    return np.array([
        w1 + 2 * w2,
        3 * w1 - w2
    ])
    
w = np.array([1.0, 2.0])
delta_w = np.array([0.01, 0.0])

c_before = phi(w)
c_after = phi(w + delta_w)

delta_c = c_after - c_before

# --------------------------------------------------------------
J = np.array([
    [1.0, 2.0],
    [3.0, -1.0]
])

delta_c_predicted = J @ np.array([0.01, 0.0]) # @ means matrix multiplication 矩陣相乘

print("Predicted delta_c:", delta_c_predicted)
print("Actual:", np.round(delta_c, 5))
print("Match:", np.allclose(delta_c_predicted, delta_c))

# --------------------------------------------------------------------------------
# 比較向量範數
latent_change = np.linalg.norm(delta_w)
hilbert_change = np.linalg.norm(delta_c)

print("Latent change:", round(latent_change, 6))
print("Hilbert change:", round(hilbert_change, 6))
print("Ratio:", round(hilbert_change / latent_change, 4))
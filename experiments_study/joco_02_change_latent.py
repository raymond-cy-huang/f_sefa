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

print("Latent Change:", delta_w)
print("Change in Hilbert Coefficients:", np.round(delta_c, 4))
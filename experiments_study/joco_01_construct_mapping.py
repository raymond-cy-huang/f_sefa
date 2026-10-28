import numpy as np


def phi(w):
    w1, w2 = w
    return np.array([
        w1 + 2 * w2, 
        3 * w1 - w2
    ])
    
w = np.array([1.0, 2.0])
c = phi(w)

print("Latent w:", w)
print("Hilbert Coefficients c:", c)

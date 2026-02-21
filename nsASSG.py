import numpy as np
def tanh(x):
    return np.tanh(x)
i1 = 0.05
i2 = 0.10
X = np.array([i1, i2])
np.random.seed()
W1 = np.random.uniform(-0.5, 0.5, (2, 2))
W2 = np.random.uniform(-0.5, 0.5, (2, 2))
b1 = 0.5  
b2 = 0.7   
net_h = np.dot(X, W1) + b1
out_h = tanh(net_h)

net_o = np.dot(out_h, W2) + b2
out_o = tanh(net_o)

print("Hidden layer output:", out_h)
print("Final network output:", out_o)
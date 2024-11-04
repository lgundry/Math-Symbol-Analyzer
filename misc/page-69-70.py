import numpy as np
from sklearn.datasets import make_classification
x,y = make_classification(n_samples=10000, weights=(0.9,0.1))
print(x.shape)
print(len(np.where(y == 0)[0]))
print(len(np.where(y==1)[0]))
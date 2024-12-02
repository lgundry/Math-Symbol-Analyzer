import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Test if .dot() works
print(A.dot(B))  # Should print: [[19 22] [43 50]]

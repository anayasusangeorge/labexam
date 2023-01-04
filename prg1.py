import numpy as np
from numpy import array
from scipy.linalg import svd
mat1 = np.array([21, 33, 67, 56])
mat2 = np.array([78, 3, 89, 77])
print("Addition")
print(np.add(mat1, mat2))
print("Subtraction")
print(np.subtract(mat1, mat2))
print("multiplication")
print(np.multiply(mat1, mat2))
print("Division")
print(np.divide(mat1, mat2))
A = array([[20, 43, 55], [23, 30, 40], [53, 42, 66]])
U, s, VT = svd(A)
print("Decomposed Matrix:\n", U)
print("inverse Matrix:\n", s)
print("Transpose Matrix:\n", VT)

import numpy as np

#  创建一个矩阵A 
A = np.array([[1,2,3],[4,5,6],[7,8,9]])

# 执行奇异值分解
U,s,V = np.linalg.svd(A,full_matrices=True)
# s[-1]=0
print("左奇异矩阵 U:\n",U)
print("奇异值 s:\n",s)
print("右奇异矩阵 V^T:\n",V)

_A = U@np.diag(s)@V
print(_A)
x = np.random.randn(3,3)
print(x@A)
print(x@_A)
loss = (x@A - x@_A).sum()
print(loss)
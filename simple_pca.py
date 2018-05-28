from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import numpy
import matplotlib.pyplot as plt

# define a matrix
A = array([[1, 2, 4, 5], [3, 4, 1 , 2]])
#print(A)
# calculate the mean of each column
M = mean(A.T, axis=1)
#print(M)
# center columns by subtracting column means
C = A - M
print(C)
# calculate covariance matrix of centered matrix
V = cov(C.T)
#print(V)
# eigendecomposition of covariance matrix
values, vectors = eig(V)
#print(vectors)
#print(values)
idx = values.argsort()[::-1]   
eigenValues = values[idx]
eigenVectors = vectors[:,idx]
print("-----------eigenVectors---------")
print (eigenVectors)
vect=eigenVectors[:,range(0,2)]
print("-----------vect---------")
print(vect)
Vecttr=numpy.transpose(vect)
print("-----------Vecttr---------")
print(Vecttr)
CT=numpy.transpose(C)
print("-----------CT---------")
print(CT)
final_data=Vecttr.dot(CT)
print("-----------final_data---------")
print(final_data)

plt.scatter(final_data[0], final_data[1], label='Class 1', c='red')

#print(eigenValues)
#print(eigenVectors)
import numpy as np
import time
from random import randint

n = 12

A = np.zeros(shape=(n, n))
B = np.zeros(shape=(n, n))
C = np.zeros(shape=(n, n))

def createMatrix():
    for i in range(0, n):
        for j in range(0, n):
            A[i][j] = randint(0, 1)
            B[i][j] = randint(0, 1)
            # A[i][j] = i*j
            # B[i][j] = 1

def matmult(A, B):
    C = np.zeros(shape=(n, n))

    for i in range(0, n):
        for j in range(0, n):
            sum = 0
            for k in range(0, n):
                sum += A[i][k] * B[k][j]
            C[i][j] = sum
    return C
    


createMatrix()

print 'Matrix A \n{}\n'.format(A)
print 'Matrix B \n{}\n'.format(B)

start = time.time()
C = matmult(A, B)
end = time.time()

timeInSeconds = end - start

print 'Result \n{}'.format(C)

print '\nTime elapsed: {}[s]\n'.format(timeInSeconds)

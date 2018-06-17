import numpy as np
import time
from random import randint

n = 3

A = np.zeros(shape=(n, n))
B = np.zeros(shape=(n, n))
C = np.zeros(shape=(n, n))

def createMatrix():
    for i in range(0, n):
        for j in range(0, n):
            A[i][j] = randint(0, 10)
            B[i][j] = randint(0, 10)


createMatrix()

print 'Matrix A \n{}\n'.format(A)
print 'Matrix B \n{}\n'.format(B)

start = time.time()
C = np.matmul(A, B)
end = time.time()

timeInSeconds = end - start

print 'Result \n{}'.format(C)

print '\nTime elapsed: {}[s]\n'.format(timeInSeconds)
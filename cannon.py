# mpiexec -n 9 --hostfile machinefile python cannon.py

import sys
import math
import numpy as np
from mpi4py import MPI
from random import randint
import time

master = 0
communicator = MPI.COMM_WORLD
numberOfProccess = communicator.Get_size()
processId = communicator.Get_rank()

c = 0

n = 16 # size array
A = np.zeros(shape=(n, n))
B = np.zeros(shape=(n, n))
C = np.zeros(shape=(n, n))
a = np.zeros(shape=(n, n))
b = np.zeros(shape=(n ,n))

def createMatrix():
    for i in range(0, n):
        for j in range(0, n):
            A[i][j] = randint(0, 10)
            B[i][j] = randint(0, 10)
            # A[i][j] = i*j
            # B[i][j] = 1
    
if processId == master:
    createMatrix()

    print "First matrix {}\n".format(A)
    print "Second matrix {}\n".format(B)

periods = [1, 1]
dims = [n, n]
coords = [None, None]
right = 0
left = 0
down = 0
up = 0

start = time.time()
cartComm = communicator.Create_cart(dims, periods, True)
cartComm.Scatter(A, a, root=master)
cartComm.Scatter(B, b, root=master)
rank = cartComm.Get_rank()
coords = cartComm.Get_coords(rank)
left, right = cartComm.Shift(1, coords[0])
up, down = cartComm.Shift(0, coords[1])
cartComm.Sendrecv_replace(a, left, 11, right, 11, None)
cartComm.Sendrecv_replace(b, up, 11, down, 11, None)
c += a[0][0] * b[0][0]

# print '{} -> A:{} B:{} -> rank: {}'.format(processId, a[0][0], b[0][0], rank)
# print 'coords: {}'.format(coords)
# print 'left: {} right: {}'.format(left, right)
# print 'c: {}'.format(c)

cartComm.Barrier()

for i in range(1, n):
    left, right = cartComm.Shift(1, 1)
    up, down = cartComm.Shift(0, 1)
    # print 'left: {} right: {}'.format(left, right)
    cartComm.Sendrecv_replace(a, left, 11, right, 11, None)
    cartComm.Sendrecv_replace(b, up, 11, down, 11, None)
    c += a[0][0] * b[0][0]
    cartComm.Barrier()

cartComm.Barrier()

C = cartComm.gather(c, root=master)
cartComm.Barrier()

end = time.time()

timeInSeconds = end - start

if processId == master:
    for i, c in enumerate(C):
        print int(c),
        if (i + 1) % n == 0:
            print '\n'
    
if processId == master:
    print '\nTime elapsed: {}[s]\n'.format(timeInSeconds)
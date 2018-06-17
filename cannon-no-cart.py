import sys
import math
import numpy as np
from mpi4py import MPI
from random import randint
from math import sqrt

master = 0
communicator = MPI.COMM_WORLD
numberOfProccess = communicator.Get_size()
processId = communicator.Get_rank()

n = int(sqrt(numberOfProccess))

A = np.zeros(shape=(n, n))
B = np.zeros(shape=(n, n))
C = np.zeros(shape=(n, n))
a = 0
b = 0
c = 0

def createMatrix():
    for i in range(0, n):
        for j in range(0, n):
            A[i][j] = randint(0, 1)
            B[i][j] = randint(0, 1)
            # A[i][j] = i*j
            # B[i][j] = 1
    
if processId == master:
    createMatrix()

    print "First matrix {}\n".format(A)
    print "Second matrix {}\n".format(B)

coords = [None, None]
right = 0
left = 0
down = 0
up = 0

A = np.asarray(A).reshape(-1)
B = np.asarray(B).reshape(-1)

if processId == master:
    start = MPI.Wtime()

if processId != master:
    A = None
    B = None

a = communicator.scatter(A, root=master)
b = communicator.scatter(B, root=master)

# communicator.Barrier()

coords[0] = math.floor(processId / n)
coords[1] = processId % n

left = (processId - coords[0] + n) if math.floor((processId - coords[0]) / n) < coords[0] else (processId - coords[0])
right = (processId + coords[0] - n) if math.floor((processId + coords[0]) / n) > coords[0] else (processId + coords[0])

up = (processId - coords[1] * n + n * n) if (processId - coords[1] * n) < 0 else (processId - coords[1] * n)
down = (processId + coords[1] * n - n * n) if (processId + coords[1] * n) >= n * n else (processId + coords[1] * n)

	
a = communicator.sendrecv(a, left, 0, None, right)
# communicator.ibsend(a, left, 0)
# a = communicator.recv(None, right)

b = communicator.sendrecv(b, up, 0, None, down)
# communicator.ibsend(b, up, 0)
# b = communicator.recv(None, down)

c += a * b

# communicator.Barrier()

right = (processId + 1 - n) if math.floor((processId + 1) / n) > math.floor(processId / n) else (processId + 1)
left = (processId - 1 + n) if math.floor((processId - 1) / n) < math.floor(processId / n) or (processId - 1 < 0) else (processId - 1)
down = (processId + n - n * n) if (processId + n) >= n * n else (processId + n)
up = (processId - n + n * n) if (processId - n) < 0 else (processId - n)

# communicator.Barrier()

for i in range(1, n):
    a = communicator.sendrecv(a, left, 0, None, right)
    # communicator.ibsend(a, left, 0)
    # a = communicator.recv(None, right)

    # communicator.Barrier()

    b = communicator.sendrecv(b, up, 0, None, down)
    # communicator.ibsend(b, up, 0)
    # b = communicator.recv(None, down)

    # communicator.Barrier()
    
    c += a * b

# communicator.Barrier()

C = communicator.gather(c, root=master)
# communicator.Barrier()

if processId == master:
    end = MPI.Wtime()
    timeInSeconds = end - start

if processId == master:
    for i, c in enumerate(C):
        print int(c),
        if (i + 1) % n == 0:
            print '\n'
    
if processId == master:
    print '\nTime elapsed: {}[s]\n'.format(timeInSeconds)

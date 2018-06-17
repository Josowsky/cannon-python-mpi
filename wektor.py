# mpiexec -n 2 --hostfile machinefile python project.py

import sys
import math
import numpy as np
from mpi4py import MPI
from random import randint

master = 0
communicator = MPI.COMM_WORLD
numberOfProccess = communicator.Get_size()
processId = communicator.Get_rank()

n = 16 # size array

A = np.empty(n*n, dtype='i')
B = np.empty(n*n, dtype='i')
C = np.empty(n*n, dtype='i')
cFromP = 0
currentElementFromA = np.empty(n, dtype='i')
currentElementFromB = np.empty(n, dtype='i')
tempA = np.empty(n, dtype='i')
tempB = np.empty(n, dtype='i')

def createMatrix():
    for i in range(0, n):
        for j in range(0, n):
            A[i*n+j] = randint(0, 10)
            B[i*n+j] = randint(0, 10)

if processId == master:
    createMatrix()
    print "First matrix {}\n".format(A)
    print "Second matrix {}\n".format(B)

# input matrices are now available, moving to initial setup stage
# giving each processing element P(i,j), the elements A(i,j) and B(i,j)
communicator.Scatter(A, currentElementFromA, root=0)
communicator.Scatter(B, currentElementFromB, root=0)

# initial setup is now complete
# proceeding towards intermediate setup
# performing shifts: Row-wise for A, Col-wise for B

# # shifts for A:
row = math.floor(processId / n)
destinationRow = (processId - row + n) if math.floor((processId - row) / n) < row else (processId - row)
sourceRow = (processId + row - n) if math.floor((processId + row) / n) > row else (processId + row)
communicator.Ibsend(currentElementFromA, destinationRow, processId)
communicator.Recv(tempA, sourceRow, sourceRow)
currentElementFromA = tempA

# #shifts for B
col = processId % n
destinationCol = (processId - col * n + n * n) if (processId - col * n) < 0 else (processId - col * n)
sourceCol = (processId + col * n - n * n) if (processId + col * n) >= n * n else (processId + col * n)
communicator.Ibsend(currentElementFromB, destinationCol, processId)
communicator.Recv(tempB, sourceCol, sourceCol)
currentElementFromB = tempB

# testing A and B:
print "processId: {}, currentElementFromA: {}, currentElementFromB: {}".format(processId, currentElementFromA[0], currentElementFromB[0])

# communicator.Barrier()

# Pseudocode
# row i of matrix a is circularly shifted by i elements to the left.
# col j of matrix b is circularly shifted by j elements up.
# Repeat n times:
#     p[i][j] multiplies its two entries and adds to running total.
#     circular shift each row of a 1 element left
#     circular shift each col of b 1 element up

sourceA = (processId + 1 - n) if math.floor((processId + 1) / n) > math.floor(processId / n) else (processId + 1)
destinationA = (processId - 1 + n) if math.floor((processId - 1) / n) < math.floor(processId / n) or (processId - 1 < 0) else (processId - 1)
sourceB = (processId + n - n * n) if (processId + n) >= n * n else (processId + n)
destinationB = (processId - n + n * n) if (processId - n) < 0 else (processId - n)

communicator.Barrier()

# Circular shifts, dimension times
for i in range(0, n):
    cFromP += currentElementFromA[0] * currentElementFromB[0]

    if processId == master:
        print "Iteration: {}".format(i+1)

    # ring-rotate A 1 time leftwards
    communicator.Ibsend(currentElementFromA, destinationA, processId)
    communicator.Recv(tempA, sourceA, sourceA)
    print "processId: {}, tempA: {}".format(processId, tempA[0])
    
    # ring-rotate B 1 time upwards
    communicator.Ibsend(currentElementFromB, destinationB, processId+n*n)
    communicator.Recv(tempB, sourceB, sourceB+n*n)
    print "processId: {}, tempB: {}".format(processId, tempB[0])

    communicator.Barrier()

communicator.Barrier()
C = communicator.gather(cFromP, root=0)

communicator.Barrier()

if processId == master:
    print "Result: \n"
    for i in range(0, n):
        print "\n"
        for j in range(0,n):
            print "{} ".format(C[i * n + j]),
import numpy as np

def getBase(numWorkers):
    base = 1
    _numWorkers = 1
    while True:
        if _numWorkers == numWorkers:
            return base
        elif _numWorkers > numWorkers:
            return None
        else:
            base += 1
            _numWorkers += base

def getMyChunk(numJobs,numWorkers,rank):
    """Returns number of events assigned to the slave calling this function."""
    assert(numJobs >= numWorkers)
    allJobs = np.arange(numJobs)
    base = getBase(numWorkers)
    jobChunks = np.array_split(allJobs, base)
    numChunks = len(jobChunks)
    counter = 0
    for i in range(numChunks):
        for j in np.arange(i,numChunks):
            if counter == rank:
                return jobChunks[i], jobChunks[j]
            counter += 1

def getMyUnfairShare(numJobs,numWorkers,rank):
    """Returns number of events assigned to the slave calling this function."""
    assert(numJobs >= numWorkers)
    allJobs = np.arange(numJobs)
    jobChunks = np.array_split(allJobs,numWorkers)
    myChunk = jobChunks[rank]
    myJobs = allJobs[myChunk[0]:myChunk[-1]+1]
    return myJobs

def varNorm(V):
# variance normalization, each image has mean 0, variance 1
    # This shouldn't happen, but zero out infinite pixels
    V[np.argwhere(V==np.inf)] = 0

    mean = np.mean(V)
    std = np.std(V)

    if std == 0:
        return np.zeros_like(V)
    V1 = (V-mean)/std

    if np.isnan(np.sum(V1)) == True: embed()
    assert np.isinf(np.sum(V1)) == False

    return V1

def donutMask(N,M,R,r):
    """
    Calculate a donut mask.

    N,M - The height and width of the image
    R   - Maximum radius from center
    r   - Minimum radius from center

    """
    centerY = N/2.
    centerX = M/2.
    mask = np.zeros((N,M))
    for i in range(N):
        xDist = i-centerY
        xDistSq = xDist**2
        for j in range(M):
            yDist = j-centerX
            yDistSq = yDist**2
            rSq = xDistSq+yDistSq
            if rSq <= R**2 and rSq >= r**2:
                mask[i,j] = 1
    return mask

from scipy.sparse.linalg import eigsh
import scipy.sparse
from numpy.matlib import repmat
def diffusionMap(myD, sigmaK, alpha=1, numEigs=16):
    # dense laplacian
    K = np.exp(-np.square(myD/sigmaK)) # kernel

    p = np.matrix(np.sum(K,axis=0))
    P = np.multiply(p.transpose(),p)
    if alpha == 1:
        K1 = K/P
    else:
        K1 = K/np.power(P,alpha)
    v = np.sqrt(np.sum(K1,axis=0))
    A = K1/np.multiply(v.transpose(),v)

    if numEigs+1 > myD.shape[0]: numEigs = myD.shape[0] - 2 
    s,u = eigsh(scipy.sparse.coo_matrix(A), k=numEigs+1, which='LM')
    u = np.real(u)
    u = np.fliplr(u)
    s = s[::-1]
    s = s[1::]
    U = u/repmat(np.matrix(u[:,0]).transpose(),1,numEigs+1)
    Y = U[:,1:numEigs+1]
    return Y,s

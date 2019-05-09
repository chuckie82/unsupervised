import h5py
import numpy as np
import os
from mpi4py import MPI
import matplotlib.pyplot as plt
import scipy.ndimage
import utils
import time

stime = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>=1, 'Require at least one mpi ranks'
assert utils.getBase(size), 'Can not deal with this size'
status = MPI.Status()

doPlot = True
numImgs = 3599
indX, indY = utils.getMyChunk(numImgs,size,rank)

# input
tic = time.time()
dir = "/reg/d/psdm/cxi/cxitut13/scratch/yoon82/simulation/6nyf"
pdb = "6NYF"
symmetry=6
nPix = 384
fname = pdb+"_signoise40.h5"
f = h5py.File(os.path.join(dir,fname),"r")
ind = None
onDiagonal = np.array_equal(indX,indY)
if onDiagonal:
    ind = indX
else:
    ind = np.concatenate((indX,indY))
numChunkImgs = len(ind)

imgStack = np.zeros((numChunkImgs, nPix, nPix),dtype=np.float32)
for i, val in enumerate(ind):
    imgStack[i,:,:] = f['/MDF/images/'+str(val)+'/image']
f.close()
print "Done loading data: ", rank, time.time() - tic

# normalization
tic = time.time()
mask = utils.donutMask(nPix,nPix,150,0)
normImgs = np.zeros((numChunkImgs,nPix*nPix))
for i in range(numChunkImgs):
    normImgs[i,:]=utils.varNorm(imgStack[i,:,:]*mask).flatten()
print "Done normalizing data: ", rank, time.time() - tic

# L2-norm
tic = time.time()
from sklearn.metrics.pairwise import euclidean_distances
myD = None
if onDiagonal:
    myD = euclidean_distances(normImgs, normImgs)
else:
    numX = len(indX)
    myD = euclidean_distances(normImgs[0:numX,:], normImgs[numX::,:])
print "Done L2-norm: ", rank, time.time() - tic
# Save distance chunks to file
dname = 'L2_' + str(rank) + '.npz'
np.savez(dname, chunkD=myD, indX=[indX[0],indX[-1]], indY=[indY[0],indY[-1]])

# Scan eps
tic = time.time()
logEps = np.linspace(-10.0, 20.0, num=20)
logSampleDistance = logEps[1]-logEps[0]
eps = np.exp(logEps)
L = np.zeros_like(logEps)
for i,e in enumerate(eps):
    K = np.exp( -np.square(myD)/ (2*e) ) # kernel
    L[i] = np.log(np.sum(K)+1e-10)
data = comm.gather(L, root=0)

# Initialize broadcast variables
numEigs = 16
if numEigs+1 > numImgs: numEigs = numImgs - 2
Y = np.zeros((numImgs,numEigs))
s = np.zeros((numEigs,))
direction = 1

if rank == 0:
    # Normalize the curve
    normL = L - np.min(L) # min is 0
    normL /= np.max(normL) # max is 1

    # Optimize eps
    gradients=np.gradient(normL)
    maxInd=np.argmax(gradients)
    if maxInd < len(gradients)-1:
        gradient = (normL[maxInd+1]-normL[maxInd]) / logSampleDistance
        # Calculate when it will reach 0.75
        targetY = 0.75
        sigmaK = np.exp((targetY - normL[maxInd]) / gradient + logEps[maxInd]) # gradient is in log space
    print "Done optimizing sigmaK: ", time.time() - tic

    # Load distance chunks
    myD = np.zeros((numImgs,numImgs))
    for i in range(size):
        dname = 'L2_' + str(i) + '.npz'
        npz = np.load(dname)
        sx = npz['indX'][0]
        ex =npz['indX'][-1]
        sy = npz['indY'][0]
        ey = npz['indY'][-1]
        myD[sx:ex+1,sy:ey+1]=npz['chunkD']

    # Symmetrize
    myD = np.maximum(myD, myD.transpose())

    # Diffusion map
    tic = time.time()
    Y,s = utils.diffusionMap(myD, sigmaK=sigmaK, numEigs=numEigs)
    print "Done diffusion map: ", time.time() - tic

    # Rotating images clockwise
    tic = time.time()
    subset = numChunkImgs
    if subset > 100: subset = 100
    corrImgStack = np.zeros((subset, nPix, nPix),dtype=np.float32)
    ang = np.zeros(subset)
    for i in range(subset):
        ang[i] = np.arctan2(Y[i,1], Y[i,0]) * (180/np.pi)
        corrImgStack[i,:,:] = scipy.ndimage.rotate(imgStack[i,:,:], (ang[i]/symmetry), reshape=False)
    fomCW=np.mean(np.var(corrImgStack,axis=0))

    # Rotating images anti-clockwise
    corrImgStack1 = np.zeros((subset, nPix, nPix),dtype=np.float32)
    for i in range(subset):
        ang[i] = np.arctan2(Y[i,1], Y[i,0]) * (180/np.pi)
        corrImgStack1[i,:,:] = scipy.ndimage.rotate(imgStack[i,:,:], (-ang[i]/symmetry), reshape=False)
    fomACW=np.mean(np.var(corrImgStack1,axis=0))
    if fomCW > fomACW:
        direction = -1
    del corrImgStack1
    print "Done determine direction: ", time.time() - tic

comm.Bcast(np.ascontiguousarray(Y), root=0)
comm.Bcast(np.ascontiguousarray(s), root=0)
comm.bcast(direction, root=0)

if doPlot and rank == 3:
    plt.plot(s, 'x-');
    plt.title('Eigenvalues');
    plt.show()
    fig, axs = plt.subplots(1, 1)
    plt.plot(Y[:, 0], Y[:, 1], '.')
    axs.set_aspect('equal', 'box')
    plt.title("Diffusion map manifold")
    plt.xlabel("eig1")
    plt.ylabel("eig2")
    plt.show()

meanImg = np.zeros((nPix,nPix), dtype=imgStack.dtype)
if rank < utils.getBase(size):
    tic = time.time()
    offset = 0
    if not onDiagonal: offset = len(indX)
    for i in range(len(indY)):
        ang = np.arctan2(Y[indY[0]+i,1], Y[indY[0]+i,0]) * (180/np.pi)
        meanImg += scipy.ndimage.rotate(imgStack[i+offset,:,:], (direction * ang/symmetry), reshape=False)
    meanImg /= len(indY)
    print "Done rotating back data: ", time.time() - tic

recvbuf=None
if rank == 0:
    recvbuf = np.empty([size, nPix, nPix], dtype=meanImg.dtype)
comm.Gather(meanImg, recvbuf, root=0)

if rank == 0:
    meanImg = np.mean(recvbuf[0:utils.getBase(size),:,:], axis=0)
    print "Time to solution: ", time.time() - stime
    # Show reconstruction
    plt.figure(figsize=(20,20))
    plt.imshow(meanImg,interpolation='none',vmin=2500); plt.show()
    fout = pdb+"_meanImag.npy"
    np.save(fout, meanImg)


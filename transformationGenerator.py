import numpy as np
import itertools
import torch
from sklearn import random_projection


izip = itertools.zip_longest
chain = itertools.chain.from_iterable
compress = itertools.compress
def rwhPrimes2(n):
    """ Input n>=6, Returns a list of primes, 2 <= p < n """
    zero = bytearray([False])
    size = n//3 + (n % 6 == 2)
    sieve = bytearray([True]) * size
    sieve[0] = False
    for i in range(int(n**0.5)//3+1):
      if sieve[i]:
        k=3*i+1|1
        start = (k*k+4*k-2*k*(i&1))//3
        sieve[(k*k)//3::2*k]=zero*((size - (k*k)//3 - 1) // (2 * k) + 1)
        sieve[  start ::2*k]=zero*((size -   start  - 1) // (2 * k) + 1)
    ans = [2,3]
    poss = chain(izip(*[range(i, n, 6) for i in (1,5)]))
    ans.extend(compress(poss, sieve))
    return ans

class transformationGenerator():
    def __init__(self, numberOfTransformations, embeddingDimension, projectionDim,numberOfEpochs, use_rp):
        self.use_rp = use_rp
        self.numberOfTransformations = numberOfTransformations
        self.embeddingDimension = embeddingDimension
        self.projectionDim = projectionDim
        self.numberOfEpochs = numberOfEpochs
        self.transforms = []
        self.reInitCounter = -1
        self.generateSeedMatrix()
        self.initializeTransforms()
        
        
    def generateSeedMatrix(self):
        temp_seed_values = np.array(list(set(rwhPrimes2(10000000)).difference(set(rwhPrimes2(100000))))) #There are 654987 prime number in this list.
        temp_seed_values = temp_seed_values[: self.numberOfTransformations * self.numberOfEpochs]
        self.seed_values = np.reshape(temp_seed_values, (self.numberOfTransformations,-1))

        
    def getGaussianRandomProjection(self,ind):
        rp = random_projection.GaussianRandomProjection(n_components=self.projectionDim,random_state=self.seed_values[ind][self.reInitCounter])
        rp.fit(np.zeros((1, self.embeddingDimension)))
        rp =  torch.Tensor(rp.components_)
        rp = rp.cuda()
        rp.requires_grad = False
        return rp
        
    def getDiagonalScaling(self):
        ds = torch.randn(self.embeddingDimension)
        ds = torch.Tensor(torch.diag(ds))
        ds = ds.cuda()
        ds.requires_grad= False
        return ds

                
    def initializeTransforms(self):
        if self.use_rp:
            for ind in range(self.numberOfTransformations):
                self.transforms.append(self.getGaussianRandomProjection(ind))
            
        else:
            for ind in range(self.numberOfTransformations):
                self.transforms.append(self.getDiagonalScaling())
           
        self.reInitCounter = self.reInitCounter + 1

    def getTransformations(self, reInit=False):
        if(reInit):
            self.initializeTransforms()
        
        return self.transforms
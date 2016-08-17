import numpy as np
import coordinate as cr 
from scipy.stats import multivariate_normal
import coordinate as cr 
import matplotlib.pyplot as plt
def LowerTriangularToVector(matrix):
    '''
    Transform a matrix's Lowertriangular part without diagnoal part to vector
    '''
    n=matrix.shape[0]
    Crr_Indice=np.tril_indices(n,-1)
    matrixVector=matrix[Crr_Indice] #lower triangular without diagnoal
    return matrixVector
def LowerTriangularVectorToSymmetricMatrix(vector,d):
    '''
    Transform a vector to a matrix LowerTriangular part and let it be symmetric
    d is the dimension to destination matrix
    '''
    Crr_Indice=np.tril_indices(d,-1)
    Matrix=np.zeros((d,d))
    Matrix[Crr_Indice]=vector
    Matrix=Matrix+Matrix.T
    return Matrix

def CovarianceMatrix(DistanceMatrix,parameter):
    '''
    Input A DistanceMatrix and sigma and l,
    And output the CovarianceMatrix of the distance 
    Cov(d1,d2)=sigma^2 exp(-|d1-d2|^2/2l)
    '''
    sigma=parameter[0]
    l=parameter[1]#should equal to the average distance
    num_people=DistanceMatrix.shape[0]

    DistanceVector=LowerTriangularToVector(DistanceMatrix)
    num_Distance=DistanceVector.size

    DV=DistanceVector.reshape(num_Distance,1)
    oneMatrix=np.ones((num_Distance,1))
    DifferenceMatrix=oneMatrix.dot(DV.T)-DV.dot(oneMatrix.T)
    #DifferenceMatrix=DifferenceMatrix**2/l# square distance
    #DifferenceMatrix=abs(DifferenceMatrix/(2*l))#abs distance
    CovarianceMatrix=(sigma**2)*np.exp(-(DifferenceMatrix**2)/l) #Covariance matrix sigma^2*exp(-|d-d`|^2/2l)
    K=CovarianceMatrix
    K = K + np.eye(K.shape[0]) * 1e-7
    return K
'''Test code
a=np.array((1,2,3,4,5,6,7,8,9))
a.shape=(3,3)
c=CovarianceMatrix(a,1)
print(c)
'''
def GPlikelihood():
    pass
def InitialGP(DistanceMatrix,parameter=np.array((1,1))):
    '''
    #The cholesky decomposition for corvarianceMatrix
    #Sigma=L%*%L.T cholesky decomposition
    #we want get sample~MVN(0,Sigma)
    #so get sample~MVN(0,Identity(n))%*%L 
    '''
    num_people=DistanceMatrix.shape[0]
    cov=CovarianceMatrix(DistanceMatrix,parameter)
    n=cov.shape[0]
    cho=np.linalg.cholesky(cov)
    sample=multivariate_normal.rvs(np.zeros(n),np.identity(n))
    sample=sample.dot(cho)
    return sample
def SampleGP(mean,Cholesky):
    '''
    Transform sample from MVN to BetaMatrix format
    Care the interface of cholesky decomposition
    '''
    n=mean.size
    sample=multivariate_normal.rvs(np.zeros(n),np.identity(n))
    GP=mean+np.dot(Cholesky,sample)
    return GP

def logGaussianProcessPrior(functionvalue,CovarianceMatrix,Cholesky,CholeskyInv):
    '''
    return the prior function value of GaussianProcess
    '''
    N=functionvalue.size
    Constant=-N/2+np.log(2*np.pi)
    LogDeterminant=-1/2*np.sum(np.log(np.square(np.diag(Cholesky))))
    Quadratic=-1/2*functionvalue.reshape(1,N).dot(CholeskyInv.dot(CholeskyInv.T)).dot(functionvalue.reshape(N,1))
    return Constant+LogDeterminant+Quadratic

def BetaMatrixPlot(DistanceMatrix,BetaMatrixTumple,i):
    '''
    plot the function distance->infect rate
    or called "kernel function"
    '''
    plt.clf()
    color=['r','g','b']
    vectorDistance=LowerTriangularToVector(DistanceMatrix)
    indices=np.argsort(vectorDistance)
    plt.gca().set_color_cycle(['red', 'green', 'blue'])
    for j in range(i):
        vectorBeta=LowerTriangularToVector(BetaMatrixTumple[j])
        plt.plot(vectorDistance[indices],vectorBeta[indices],'k',color=color[j])
    plt.show()
def generalBetaMatrixPlot(DistanceMatrix,BetaMatrix):
    plt.clf()
    vectorDistance=LowerTriangularToVector(DistanceMatrix)
    indices=np.argsort(vectorDistance)
    plt.plot(vectorDistance[indices],vectorBeta[indices],'k')
def GPPlot(DistanceMatrix,recordGP):
    '''
    plot the function distance->infect rate
    or called "kernel function"
    '''
    plt.clf()
    vectorDistance=LowerTriangularToVector(DistanceMatrix)
    indices=np.argsort(vectorDistance)
    i = recordGP.shape[0]
    for j in range(i):
        vectorBeta=recordGP[j,:]
        plt.plot(vectorDistance[indices],vectorBeta[indices],'k')
    plt.show()
    plt.show()
def kernelFunctonPlot(DistanceMatrix,recordGP,record,method):
    plt.clf()
    iterNa=recordGP.shape[0]
    vectorDistance=LowerTriangularToVector(DistanceMatrix)
    indices=np.argsort(vectorDistance)
   
    for j in range(iterNa):
        BetaMatrix=cr.BetaMatrix(DistanceMatrix,record[j,:],method)
        BetaVectorBaseline=LowerTriangularToVector(BetaMatrix)
        BetaVector=np.exp(np.log(BetaVectorBaseline)+recordGP[j])
        plt.plot(vectorDistance[indices],BetaVectorBaseline[indices],'k',color="green")
        plt.plot(vectorDistance[indices],BetaVector[indices],'k')
    plt.show()
def kernelFunctonPlotRebuild(DistanceMatrix,recordGP,meanParameter,method,simulationParameter,InitialGP):
    plt.clf()
    iterNa=recordGP.shape[0]
    vectorDistance=LowerTriangularToVector(DistanceMatrix)
    indices=np.argsort(vectorDistance)
    BetaMatrix=cr.BetaMatrix(DistanceMatrix,meanParameter,"gradient")
    BetaVectorBaseline=LowerTriangularToVector(BetaMatrix)

    BetaMatrixSimulation=cr.BetaMatrix(DistanceMatrix,np.delete(simulationParameter,-1),method)
    BetaVectorSimulation=LowerTriangularToVector(BetaMatrixSimulation)
    plt.plot(vectorDistance[indices],BetaVectorSimulation[indices],'k',color="green")
    plt.plot(vectorDistance[indices],BetaVectorBaseline[indices],'k',color="yellow")
    '''
    maxGP=recordGP.max(0)
    BetaVector=np.exp(np.log(BetaVectorBaseline)+maxGP)
    plt.plot(vectorDistance[indices],BetaVector[indices],'k')
    minGP=recordGP.min(0)
    BetaVector=np.exp(np.log(BetaVectorBaseline)+minGP)
    plt.plot(vectorDistance[indices],BetaVector[indices],'k')
    '''
    medianGP=np.median(recordGP,0)
    BetaVector=np.exp(np.log(BetaVectorBaseline)+medianGP)
    plt.plot(vectorDistance[indices],BetaVector[indices],'k')
    upper95GP=np.percentile(recordGP,95,0)
    BetaVector=np.exp(np.log(BetaVectorBaseline)+upper95GP)
    plt.plot(vectorDistance[indices],BetaVector[indices],'k',color="red")
    lower95GP=np.percentile(recordGP,5,0)
    BetaVector=np.exp(np.log(BetaVectorBaseline)+lower95GP)
    plt.plot(vectorDistance[indices],BetaVector[indices],'k',color="blue")
    BetaVector=np.exp(np.log(BetaVectorBaseline)+InitialGP)
    plt.plot(vectorDistance[indices],BetaVector[indices],'k',color="brown")

    plt.show()
'''
a=np.array((1,2,3,4,5,6,7,8))
a.shape=(4,2)
d=cr.DistanceMatrix(a)
corre=CovarianceMatrix(d,1)
sample=SampleGP(np.zeros(corre.shape[0]),corre,4)
print(d)
print(corre)
print(sample)
'''
class GaussianProcess:
    def __init__(self,DistanceMatrix,parameter):
        '''
        parameter=[sigma,l]
        '''
        self.DistanceMarix=DistanceMatrix
        self.CovarianceMatrix=CovarianceMatrix(DistanceMatrix,parameter)
        self.Cholesky=np.linalg.cholesky(self.CovarianceMatrix)
        self.CholeskyInv=np.linalg.inv(self.Cholesky)
    def SampleForGP(self,mean):
        return SampleGP(mean,self.Cholesky)
    def GPprior(self,value):
        return logGaussianProcessPrior(value,self.CovarianceMatrix,self.Cholesky,self.CholeskyInv)
    
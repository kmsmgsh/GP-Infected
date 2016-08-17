import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import GP as gp
import time
#plt.ion()
#plt.style.use('ggplot')
class Metropolis:
    def OnestepMetropolis(self,density,theta,sigma,GP,i):
        theta_star=np.exp(np.random.normal(np.log(theta[i]), sigma))
        thetastar=theta.copy()
        thetastar[i]=theta_star
        #Accept the new beta value with the probability f(beta_start)/f(beta)
        p=min(np.exp(density(thetastar,GP)-density(theta,GP)),1)####################Transfor p/p,or use log normal:btw: p/p=1 solved
        if np.random.uniform(0,1)<p:
            #Accept the new Value
            return [1,theta_star]
            #count the number of accept
        else:
            return [0,theta[i]]
    def OnestepMetropolisGP(self,densityGP,parameter,GaussProcess,CurrentGP):
        newGP=GaussProcess.SampleForGP(CurrentGP)
        #Accept the new beta value with the probability f(beta_start)/f(beta)
        lognew=densityGP(parameter,GaussProcess,newGP)
        logold=densityGP(parameter,GaussProcess,CurrentGP)
        mid=lognew-logold
        p=min(np.exp(mid),1)####################Transfor p/p,or use log normal:btw: p/p=1 solved
        
        if np.random.uniform(0,1)<p:
            #Accept the new Value
            return [1,newGP]
            #count the number of accept
        else:
            return [0,CurrentGP]
class multiMetropolis(Metropolis):
    def __init__(self,IterNa,density,initial_parameter,sigma,initialGP,GaussianProcess,GPdensity,GPmode="c",parameterMode="m"):
        self.IterNa=IterNa
        self.initial_parameter=np.array(initial_parameter)
        self.dimension=self.initial_parameter.size
        self.density=density
        self.sigma=sigma
        self.initialGP=initialGP
        self.GaussianProcess=GaussianProcess
        self.GPmode=GPmode
        self.GPdensity=GPdensity
        self.parameterMode=parameterMode
        self.Mainprocess()
    def Mainprocess(self):
        parameter=self.initial_parameter
        record=parameter
        Accept=np.zeros((self.IterNa,self.dimension))
        GP=self.initialGP
        AcceptGP=np.zeros(self.IterNa)
        recordGP=GP
        for i in range(0,self.IterNa):
            if self.parameterMode!="c":
                '''
                if parameterMode is constant, the parameter do not move, just avoid the MH algorithm
                value is as the initialValue
                the parameter part is as the mean of GP
                '''
                for j in range(0,self.dimension):
                    result=self.OnestepMetropolis(self.density[j],parameter,self.sigma[j],GP,j)
                    Accept[i,j]=result[0]
                    parameter[j]=result[1]
            record=np.vstack((record,parameter))
            if self.GPmode!="c":
                resultGP=self.OnestepMetropolisGP(self.GPdensity,parameter,self.GaussianProcess,GP)
                AcceptGP[i]=resultGP[0]
                recordGP=np.vstack((recordGP,resultGP[1]))
                GP=resultGP[1]
        self.record=record
        self.Accept=Accept
        if self.GPmode!="c":
            self.recordGP=recordGP
            self.AcceptGP=AcceptGP
    def showplot(self,i):
        plt.clf()
        plt.plot(range(self.IterNa+1),self.record[:,i])
        plt.plot(range(200,self.IterNa),self.record[200:self.IterNa,i])
        plt.show()
        #plt.figure()
        plt.hist(self.record[:,i],bins=50)
        #plt.figure()

        plt.show()
       
    def plotcountour(self,i,j):
        plt.clf()
        plt.plot(self.record[:,i],self.record[:,j], 'ko', alpha=0.4)
        plt.show()
        plt.plot(self.record[:,i],self.record[:,j])
        plt.show()
    def printall(self,i):
        print("Accept rate is")
        print(sum(self.Accept[:,i])/self.IterNa)
        print("Mean is")
        print(np.mean(self.record[200:self.IterNa,i]))
    def printAcceptRateGP(self):
        print("Accept rate is")
        print(sum(self.AcceptGP)/self.IterNa)
    def plotOneComponentGP(self):
        plt.clf()
        plt.plot(range(self.IterNa+1),self.recordGP[:,0])
        plt.show()
    def saveResult(self,filenameGP,filenameParameter):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        np.savetxt("GP"+filenameGP+timestr+".csv", self.recordGP, delimiter=",")
        print("GP results have successfully saved!")
        np.savetxt("parameter"+filenameParameter+timestr+".csv",self.record,delimiter=",")
        print("parameter results have successfully saved!")
class GaussianProcessMetropolis(Metropolis):
    '''
    This is the kernel function with only GP update but with some prior infor as gradient(exponential)
    '''
    def _init_(self,IterNa,initialGP,Gaussianprocess,GPdensity,baseline):
        self.IterNa=IterNa
        self.GPdensity=GPdensity
        self.GaussianProcess=GaussianProcess
        self.initialGP=initialGP
        self.baseline=baseline
        self.Dimension=initialGP.size
        self.populationsize=(1+np.sqrt(1+8*self.Dimension))/2
        self.Mainprocess()
    def Mainprocess(self):
        GP=self.initialGP
        AcceptGP=np.zeros(self.IterNa)
        recordGP=GP
        for i in range(self.IterNa):
            resultGP=self.OnestepMetropolisGP(self.GPdensity,parameter,self.GaussianProcess,GP)
            AcceptGP[i]=resultGP[0]
            recordGP=np.vstack((recordGP,resultGP[1]))
        
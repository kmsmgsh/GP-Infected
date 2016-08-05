import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import GP as gp
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
        p=min(np.exp(densityGP(parameter,GaussProcess,newGP)-densityGP(parameter,GaussProcess,CurrentGP)),1)####################Transfor p/p,or use log normal:btw: p/p=1 solved
        if np.random.uniform(0,1)<p:
            #Accept the new Value
            return [1,newGP]
            #count the number of accept
        else:
            return [0,CurrentGP]
class multiMetropolis(Metropolis):
    def __init__(self,IterNa,density,initial_parameter,sigma,initialGP,GaussianProcess,GPdensity,GPmode="c"):
        self.IterNa=IterNa
        self.initial_parameter=np.array(initial_parameter)
        self.dimension=self.initial_parameter.size
        self.density=density
        self.sigma=sigma
        self.initialGP=initialGP
        self.GaussianProcess=GaussianProcess
        self.GPmode=GPmode
        self.GPdensity=GPdensity
        self.Mainprocess()
    def Mainprocess(self):
        parameter=self.initial_parameter
        record=parameter
        Accept=np.zeros((self.IterNa,self.dimension))
        GP=self.initialGP
        AcceptGP=np.zeros(self.IterNa)
        recordGP=GP
        for i in range(0,self.IterNa):
            for j in range(0,self.dimension):
                result=self.OnestepMetropolis(self.density[j],parameter,self.sigma[j],GP,j)
                Accept[i,j]=result[0]
                parameter[j]=result[1]
            record=np.vstack((record,parameter))
            if self.GPmode!="c":
                resultGP=self.OnestepMetropolisGP(self.GPdensity,parameter,self.GaussianProcess,GP)
                AcceptGP[i]=resultGP[0]
                recordGP=np.vstack((recordGP,resultGP[1]))
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
        plt.plot(self.record[:,0],self.record[:,1], 'ko', alpha=0.4)
        #plt.figure()

        plt.show()
        plt.hist(self.record[:,i],bins=50)
        #plt.figure()

        plt.show()
        plt.plot(self.record[:,0],self.record[:,1])
        plt.show()
    def printall(self,i):
        print("Accept rate is")
        print(sum(self.Accept[:,i])/self.IterNa)
        print("Mean is")
        print(np.mean(self.record[200:self.IterNa,i]))
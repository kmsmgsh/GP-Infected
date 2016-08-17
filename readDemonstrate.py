'''
This file contains the function that read the MCMC results in the current file
'''
import os
import numpy as np
import GP as gp
import coordinate as cr
a=os.listdir()
#print (a)
markGP=[s for s in a if s.find('GP')!=-1 and s.find(".csv")!=-1]
markparameter=[s for s in a if s.find('parameter')!=-1 and s.find(".csv")!=-1]

print(markGP)
print(markparameter)
print("plot which data")
k=input()#list start with 0
k=int(k)
recordGP=np.loadtxt(markGP[k],delimiter=",")
recordParameter=np.loadtxt(markparameter[k],delimiter=",")
geo=np.loadtxt("geo_uniform.txt")
DistanceMatrix=cr.DistanceMatrix(geo)
gp.kernelFunctonPlot(DistanceMatrix,recordGP,recordParameter,"gradient")
InitialGP=recordGP[0,:]
gp.kernelFunctonPlotRebuild(DistanceMatrix,recordGP,recordParameter[0,:],"gradient",recordParameter[0,:],InitialGP)
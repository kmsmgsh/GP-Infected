import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import Metropolis3 as mp3
import generator_temp_transprob as gc2
import likelihodPhi as lk2
import coordinate as cr
from scipy.stats import beta
from functools import partial
import GP as gp
population=25
model1=gc2.heteregeneousModel(population,[0.4,0.1,0.3],True,True,"gradient","uniform",False)
#model1.Animate()
estimate=lk2.Estimation(model1.record,model1.geo,method="gradient")
GPDoc=gp.GaussianProcess(estimate.DistanceMatrix,np.array((1,np.mean(estimate.DistanceMatrix))))
InitialGP=GPDoc.SampleForGP(np.zeros(population*(population-1)/2))

Metro=mp3.multiMetropolis(1000,None,[0.3,0.3,0.3],None,InitialGP,GPDoc,estimate.GaussianPriorGP,"Change","c")
gp.GPPlot(model1.DistanceMatrix,Metro.recordGP)
gp.kernelFunctonPlot(model1.DistanceMatrix,Metro.recordGP,Metro.record,"gradient")
gp.kernelFunctonPlotRebuild(model1.DistanceMatrix,Metro.recordGP,[0.3,0.3,0.3],"gradient",[0.4,0.1,0.3],InitialGP)
Metro.printAcceptRateGP()
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
#import Metropolis as mp
#import Metropolis2 as mp2
import Metropolis3 as mp3
#import generator_temp as gc
import generator_temp_transprob as gc2
#import likelihood as lk
import likelihodPhi as lk2
import coordinate as cr
from scipy.stats import beta
from functools import partial
import GP as gp
#plt.ion()
#plt.style.use('ggplot')
population=100
model1=gc2.heteregeneousModel(population,[0.4,10,0.3],False,False,"gradient","uniform",False)
#model1=gc2.heteregeneousModel(50,[5,0.2,1,0.3],True,True,"powerlaw","uniform",False)
#model1.Animate()
#estimate=lk2.Estimation(model1.record,model1.geo,method="powerlaw")
estimate=lk2.Estimation(model1.record,model1.geo,method="gradient")
#Metro=mp3.multiMetropolis(1000,[estimate.GammaPosteriorBeta0,estimate.GammaPosteriorGamma,estimate.GammaPosteriorPhi],[0.1,0.1,5],[0.5,0.5,0.4])
#Metro=mp3.multiMetropolis(1000,[partial(estimate.GammaPriorGeneralPosterior,i=0),partial(estimate.GammaPriorGeneralPosterior,i=1),partial(estimate.GammaPriorGeneralPosterior,i=2)],[0.1,0.1,5],[0.5,0.5,0.4])
#Metro=mp3.multiMetropolis(1000,[estimate.GammaPosteriorBeta0,estimate.GammaPosteriorGamma],[0.1,0.1],[0.4,0.4])
#Metro=mp3.multiMetropolis(1000,[partial(estimate.GammaPriorGeneralPosterior,i=0),partial(estimate.GammaPriorGeneralPosterior,i=1),partial(estimate.GammaPriorGeneralPosterior,i=2),partial(estimate.GammaPriorGeneralPosterior,i=3)],[3,0.1,0.9,1],[0.5,0.5,0.4,0.4])
#InitialGP=np.zeros(population*(population-1)/2)
InitialGP=gp.InitialGP(estimate.DistanceMatrix)
BetaMatrix=model1.BetaMatrix
BetaMatrix3=cr.BetaMatrix(model1.DistanceMatrix,[0.2,7])
gp.BetaMatrixPlot(model1.DistanceMatrix,[BetaMatrix,np.exp(np.log(BetaMatrix)+gp.LowerTriangularVectorToSymmetricMatrix(InitialGP,BetaMatrix.shape[0])),BetaMatrix3],3)
Metro=mp3.multiMetropolis(1000,[partial(estimate.GammaPriorGeneralPosterior,i=0),partial(estimate.GammaPriorGeneralPosterior,i=1),partial(estimate.GammaPriorGeneralPosterior,i=2)],[0.1,0.9,1],[0.7,0.5,0.7],InitialGP)
Metro.showplot(0)
Metro.printall(0)
Metro.showplot(1)
Metro.printall(1)
Metro.showplot(2)
Metro.printall(2)
Metro.showplot(3)
Metro.printall(3)
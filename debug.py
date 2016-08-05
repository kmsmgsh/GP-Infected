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
'''
geo=cr.geodata(50)
geo=cr.geodata(50,"uniform",xbound=100.0,ybound=100.0,history=False)
Distance=cr.DistanceMatrix(geo)
BetaMatrix=cr.BetaMatrix(Distance,[0.3,5])
gp.BetaMatrixPlot(Distance,BetaMatrix)
'''
model1=gc2.heteregeneousModel(50,[0.3,5,0.3])
gp.BetaMatrixPlot(model1.DistanceMatrix,model1.BetaMatrix,1)

#gp.BetaMatrixPlot(model1.DistanceMatrix,model1.BetaMatrix)
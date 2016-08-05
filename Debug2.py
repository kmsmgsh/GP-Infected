import numpy as np
import generator_temp_transprob as gc
import coordinate as cr
import Metropolis3 as mp3
import likelihodPhi as llk
import GP as gp
geo=cr.geodata(40)
Distance=cr.DistanceMatrix(geo)
Corr=gp.CovarianceMatrix(Distance,np.array((1,0.01)))
Chol=np.linalg.cholesky(Corr)
print(Chol)
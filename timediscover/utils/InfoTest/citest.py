import os
os.environ['R_HOME'] = '/home/liumingzhou/Miniconda3/envs/rlanguage/lib/R' 

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
robjects.packages.importr('CItest')
#print(robjects.r['getwd']())
robjects.r('''source('./time-discovery/timediscover/utils/InfoTest/CIcodes.R')''')

def nparr_to_Rmatrix(arr):
    # convert a 2d np.array to R.matrix
    # https://stackoverflow.com/questions/4592911/convert-numpy-array-to-matrix-rpy2-kmeans
    n, d = arr.shape
    vec = robjects.FloatVector(arr.transpose().reshape((arr.size)))
    mat = robjects.r.matrix(vec, nrow=n, ncol=d)
    return mat
              
def scott_rule_of_thumb(data_z):
    '''
    Set kernel width by Scott's rule of thumb
    Args: data_z shape (n,dim_z)
    '''
    n,dim_z = data_z.shape
    kernel_width = n**(-1./(dim_z+4))
    return kernel_width

def get_cind_statistic(X,Y,Z):
    '''
    Get the conditional independence testing statistic rho(X,Y,Z)
    Args: X,Y shape both (n,), Z shape (n,dim_z)
    '''
    X,Y = robjects.FloatVector(X), robjects.FloatVector(Y)
    kernel_width = scott_rule_of_thumb(Z)
    Z = nparr_to_Rmatrix(Z)
    CITest = robjects.globalenv['CI.test']
    return CITest(X,Y,Z,kernel_width) 

def get_mind_statistic(X,Y):
    '''
    Get the marginal independence testing statistic rho(X,Y)
    Args: X,Y shape both (n,)
    '''
    X,Y = robjects.FloatVector(X), robjects.FloatVector(Y)
    MargTest = robjects.globalenv['Independence.test']
    return MargTest(X,Y) 

path = './time-discovery/timediscover/utils/InfoTest/null_distribution_samples'

def cind_test(X,Y,Z):
    '''
    Conditional independence testing X ind Y|Z using Cai et al. 2022
    Args: X,Y shape both (n,); Z shape (n,) or (n,dim_z)
    '''
    if len(Z.shape)==1:
        Z = np.expand_dims(Z,1)
    n,dim_z = Z.shape  
    
    rho = get_cind_statistic(X,Y,Z)
    null_dists = np.load(os.path.join(path,'dimz_{}.npy'.format(dim_z)))
    pvalue = (np.array(null_dists)>rho).mean()
    return pvalue

def mind_test(X,Y):
    '''
    Marginal independence testing X ind Y using Cai et al. 2022
    Args: X,Y shape both (n,)
    '''
    rho = get_mind_statistic(X,Y)
    null_dists = np.load(os.path.join(path,'dimz_0.npy'))
    pvalue = (np.array(null_dists)>rho).mean()
    return pvalue
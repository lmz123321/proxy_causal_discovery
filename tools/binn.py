import numpy as np
import pandas as pd

def uniform_bin(x,num_bins,alpha,mini=None,maxi=None):
    '''
    bin the continous X into a categorical one
    output range 1,2,...,num_bins
    todo: if mean+3sigma>x.max(), iteratively use mean+2.5sigma ...
    '''
    mini = x.mean()-alpha*x.std() if mini==None else mini
    maxi = x.mean()+alpha*x.std() if maxi==None else maxi
    
    bins = np.linspace(mini,maxi,num_bins-1)
    tx = np.digitize(x,bins)
    
    return tx+1

def quantile_bin(x,num_bins,eps=1e-5):
    '''
    bin a comtinous variable by sample quantiles, such that each bin has equal sample points
    output range: 1,2,...,num_bins
    note: numpy.digitze bin[i-1]<=x<bin[i], we enlarge the last bin a little to include x.max() to the *num_bins*-th bin
    '''
    _, bins = pd.qcut(x,num_bins,retbins=True,duplicates='drop')
    bins[-1] += eps
    tx = np.digitize(x,bins)
    
    return tx
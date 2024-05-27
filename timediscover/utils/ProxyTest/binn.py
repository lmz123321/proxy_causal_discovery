import numpy as np
import pandas as pd

def quantile_bin(x,num_bins,eps=1e-5):
    '''
    Bin a comtinous variable by sample quantiles, such that each bin has equal sample points
    Output range: 1,2,...,num_bins
    note: numpy.digitze bin[i-1]<=x<bin[i], we enlarge the last bin a little to include x.max() to the *num_bins*-th bin
    '''
    _, bins = pd.qcut(x,num_bins,retbins=True,duplicates='drop')
    bins[-1] += eps
    tx = np.digitize(x,bins)
    return tx


def uniform_bin(x,num_bins,alpha,mini=None,maxi=None):
    '''
    Bin the continous X into a categorical one
    Output range 1,2,...,num_bins
    - If given mini,maxi, we bin [mini,maxi] into num_bins subintervals
      If not given, we bin [mean-alpha*std,mean+alpha*std]
    '''
    mini = x.mean()-alpha*x.std() if mini==None else mini
    maxi = x.mean()+alpha*x.std() if maxi==None else maxi
    
    bins = np.linspace(mini,maxi,num_bins-1)
    tx = np.digitize(x,bins)
    return tx+1


def estimate_cond_cdf(valuex,valuedw,x,dw):
    '''
    Estimate F(valuex|valuew):=P(X<=valuex|W=valuew) based on dataset (x,dw)
    '''
    return (x[dw==valuedw]<=valuex).sum()/(dw==valuedw).sum()

def compute_objective(valuex,i,d,x,dw):
    '''
    Compute \sum_j=1^i (-1)^{1+j} d[j] F[valuex|w_j] based on dataset (x,dw)
    '''
    summ = 0
    for j in range(1,i+1):
        summ += (-1)**(1+j) * d[j-1] * estimate_cond_cdf(valuex,j,x,dw)
    return summ

def get_linind_bins(x,dw,lx,lw,alpha,step):
    '''
    Get Linear Independence Bins for X based on Alg-1 in the ICML paper
    Args: - x: np.array, data for x
          - dw: np.array, discretized w (obtained using uniform_bin or quantile_bin), take values in 1,...,num_binw
          - lx,lw: number bins for X,W, respectively
          - alpha: we set the first bin point to x.mean()-alpha*x.std()
          - step: searching step for x_i+1
    Note: if the generated xbins only covers a small part of rangex such that most dx=1 or lx, 
          you can set a larger alpha and a larger step
          
    Example: size = 1000
             lx,lw = 10,6
             w = np.random.normal(0,1,size)
             x = w + np.random.normal(0,1,size)
             dw = quantile_bin(w,lw)
             xbins = get_linind_bins(x,dw,lx,lw,alpha=2.5,step=.9) # set alpha,step larger if your std(x) is large
             dx = np.digitize(x,xbins)+1
             # check pWgivX has full rank
             pddata = pd.DataFrame(np.stack([dx,dw]).transpose(1,0),columns=['X','W'])
             pWgivX = condEstimate(pddata,['W'],[lw],'X',lx) # from timediscover.utils.ProxyTest.estimator import condEstimate
             print(np.linalg.matrix_rank(pWgivX))
    '''
    xbins = [x.mean()-alpha*x.std()] # init x1 with mean(x)-std(x) TODO: set this customly
    assert estimate_cond_cdf(xbins[0],1,x,dw)!=0

    for i in range(2,lw+1):
        d = list()
        for j in range(1,i+1):
            # compute the matrix F(x_[i-1]|w_[i]\j)
            rangew = list(range(1,i+1)); rangew.remove(j) 
            rangex = list(range(1,i))
            matrix = list()
            for rx in rangex:
                for rw in rangew:
                    matrix.append(estimate_cond_cdf(xbins[rx-1],rw,x,dw)) 
            matrix = np.array(matrix).reshape(len(rangex),len(rangew))
            d.append(np.linalg.det(matrix))

        # search for x_i such that \sum_j=1^i (-1)^{1+j} d[j] F[valuex|w_j] != 0
        newx = xbins[-1]+step
        while compute_objective(newx,i,d,x,dw)==0:
            newx += step
        xbins.append(newx)
    
    # set the rest lx-lw bins by uniform binning
    while len(xbins)<lx-1:
        xbins.append(xbins[-1]+step)
    return xbins
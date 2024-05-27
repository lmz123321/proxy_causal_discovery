import numpy as np
import pandas as pd

from timediscover.utils.ProxyTest.proxy import ProxyTest
from timediscover.utils.ProxyTest.binn import quantile_bin # if not matrix invertible, use the searching alg. in the paper to search for bins

def heuristic_levels(dimw):
    # return heuristic levels for X,W,Y, where levelx,levely are ints, levelw is a list of len(dimw)
    if dimw==1:
        levelw = [10]
        levelx = 16
    elif dimw==2:
        levelw = [4,4]
        levelx = 18
    elif dimw==3:
        levelw = [2,2,2]
        levelx = 12
    else:
        levelw = [2,2,2] + [1]*(dimw-3)
        levelx = sum(levelw) + 2
    levely = 5
    return levelx,levelw,levely


class ProxyCITest:
    def __init__(self,data,node_names):
        self.data = pd.DataFrame(data,columns=node_names)
        
    def __call__(self,X,Y,W,C,ratio):
        '''
        Args: X,Y causal pair of interest, both strings
              W proxies, list of strings
              C observed confounders, list of strings
              alpha: C in [mean-alpha*std,mean+alpha*std] will be kept, as a way to conditioning on C
                     the larger alpha, the more data will be kept but the weak the conditioning effect
        '''
        # condition on C
        mean,std = self.data[C].mean(), self.data[C].std()
        condition = (self.data[C]<mean+ratio*std) & (self.data[C]>mean-ratio*std)
        cond_data = self.data[condition.all(axis=1)]
        
        # renaming: X->'X', Y->'Y', W->'W1,W2,...,Wd'
        cond_data = cond_data[[X,Y]+W]
        rename_dict = {X:'X',Y:'Y'}
        for ind,w in enumerate(W):
            rename_dict[w] = 'W{}'.format(ind+1)
        cond_data.rename(columns=rename_dict, inplace=True)
        
        # discretization: we use quantile bin as default because it saves computation cost
        # if you meet P(W|X) does not have full rand (which is very rare in our practice), just use the get_linind_bins() in binn.py to get bins for X
        levelx,levelws,levely = heuristic_levels(len(W))
        cond_data['X'] = quantile_bin(cond_data['X'],levelx)
        cond_data['Y'] = quantile_bin(cond_data['Y'],levely)
        for ind in range(len(W)):
            cond_data['W{}'.format(ind+1)] = quantile_bin(cond_data['W{}'.format(ind+1)],levelws[ind])
        
        # proxy-test
        tester = ProxyTest(cond_data,levelx,levelws,levely)
        pvalue = tester.test()
        return pvalue
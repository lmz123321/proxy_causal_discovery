# This script implement marginal/conditional probabiltity estimators

import pandas as pd
import numpy as np
from itertools import product

def margEstimate(pdData,name,level,string_value=False):
    r"""
    Functs: - estimate the marginal probability vector
            - return [P(a0(1)),...,P(a0(k))] for A0 with k levels
    """
    n = pdData.shape[0]
    prb = list()
    prbDict = dict(pdData[name].value_counts(normalize=True))
    
    for ind in range(level):
        if string_value:
            key = '{}({})'.format(name.lower(),ind+1)
        else:
            key = ind+1
        prb.append(prbDict[key])

    #assert np.sum(prb) == 1, 'Marginal probability not sum to 1.'
    return prb

def condEstimate(pdData,nameX,levelX,nameC,levelC,string_value=False):
    r"""
    Functs: - estimate Pr(X1,X2,...,Xn|C)
    Args: - nameX and levelX are both list()
          - nameC and levelC are both single value, since we only face scenarios where there is one conditioned variable
          - return an array with shape(k1,k2,...,kc), where kn is the level of Xn
    value range should be 1,2,3,...,k
    """
    # check data type
    assert isinstance(nameX,list)
    assert isinstance(levelX,list)
    assert isinstance(nameC,str)
    assert isinstance(levelC,int)
    
    colX = [pdData['{}'.format(name)] for name in nameX]
    condPrb = pd.crosstab(colX, pdData[nameC], normalize='columns')
    
    prb = np.zeros(tuple(levelX+[levelC]))
    iterations = [range(level) for level in levelX] + [range(levelC)]
    for inds in product(*iterations):
        # inds from (0,0,0) to (levelx1-1,levelx2-1,...,levelc-1)
        key = list()
        for i,ind in enumerate(inds[:-1]):
            if string_value:
                name = '{}({})'.format(nameX[i].lower(),ind+1)
            else:
                name = ind+1
            key.append(name)
        if string_value:
            name = '{}({})'.format(nameC.lower(),inds[-1]+1)
        else:
            name = inds[-1]+1
        key.append(name)
        # key ('x1(i1)','x2(i2)',...,'c(j)')
        prb[inds] = condPrb[[key[-1]]].loc[tuple(key[:-1]),:].to_numpy().item()
        
    return prb
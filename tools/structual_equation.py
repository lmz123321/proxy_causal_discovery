import numpy as np
from functools import partial

def symmtric_exp(beta,size):
    '''
    symmetric exponential distribution
    '''
    randsign = 2*np.random.randint(low=0,high=2,size=size)-1
    exp = np.random.exponential(scale=beta,size=size)
    return randsign*exp

def symmetric_gamma(k,theta,size):
    '''
    symmetric gamma distribution
    '''
    randsign = 2*np.random.randint(low=0,high=2,size=size)-1
    gamma = np.random.gamma(shape=k,scale=theta,size=size)
    return randsign*gamma

def distribution(name,low=-0.25,high=0.25,std=0.1,beta=0.2,k=2.0,theta=0.1):
    # usage: distribution(name)(size=size)
    
    if name=='uniform':
        funct = partial(np.random.uniform,low=low,high=high)
    elif name=='gaussian':
        funct = partial(np.random.normal,loc=0,scale=std)
    elif name=='exponential':
        funct = partial(symmtric_exp,beta=beta)
    elif name=='gamma':
        funct = partial(symmetric_gamma,k=k,theta=theta)
    else:
        raise ValueError('Only support uniform,gaussian,exponential,gamma distribution.')
    return funct

def function(name,alpha=0.5):
    '''
    usage: y = function(name)(x)
    '''
    if name=='linear':
        funct = lambda x:x
    elif name=='sin':
        funct = lambda x:np.sin(np.pi/2*x)
    elif name=='tanh':
        funct = lambda x:np.tanh(x)
    elif name=='sqrt':
        funct = lambda x: np.sign(x)*np.sqrt(abs(x/2))
    else:
        raise ValueError('Only support linear,sin,tanh,sqrt function.')
    return lambda x:alpha*funct(x)


def rand_steq(sdag,ftdag,funcs=['linear','sin','tanh','sqrt'],dists=['uniform','gaussian','exponential','gamma']):
    '''
    generate random structual equations for each cause-effect pairs
    pick one function for each edge, one exo-dist for each vertex 
    effect = func(cause) + dist
    '''
    ssteq = dict()
    for edge in sdag.edges:
        cause,effect = edge
        key = '{}->{}'.format(cause,effect)
        ssteq[key] = np.random.choice(funcs)
        
    for node in sdag.nodes:
        key = '{}->{}'.format(node,node)
        ssteq[key] = np.random.choice(funcs)
        key = node
        ssteq[key] = np.random.choice(dists)
        
    ftsteq = dict()
    for edge in ftdag.edges:
        cause,effect = edge
        key = '{}->{}'.format(cause,effect)

        _cause = cause.split('_')[0]; _effect = effect.split('_')[0]
        _key = '{}->{}'.format(_cause,_effect)
        
        ftsteq[key] = ssteq[_key]
        
    assert len(ftsteq) == len(ftdag.edges), 'Mismatched st-eqs and edges in fulltime dag.'
    return ssteq,ftsteq
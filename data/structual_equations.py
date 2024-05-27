import numpy as np
from functools import partial

def symmtric_exp(beta,size):
    # beta large => std large
    # by default, beta=0.5 (similar to N(0,1))
    randsign = 2*np.random.randint(low=0,high=2,size=size)-1
    exp = np.random.exponential(scale=beta,size=size)
    return randsign*exp

def symmetric_gamma(k,theta,size):
    # please fix k=2, and theta large => std large
    # by default, theta=0.3 (similar to N(0,1))
    randsign = 2*np.random.randint(low=0,high=2,size=size)-1
    gamma = np.random.gamma(shape=k,scale=theta,size=size)
    return randsign*gamma

def distribution(name,std=1,beta=0.5,k=2,theta=0.3):
    # usage: distribution(name)(size=size)
    if name=='gaussian':
        funct = partial(np.random.normal,loc=0,scale=std)
    elif name=='uniform':
        funct = partial(np.random.uniform,low=-std,high=std)
    elif name=='exponential':
        funct = partial(symmtric_exp,beta=beta)
    elif name=='gamma':
        funct = partial(symmetric_gamma,k=k,theta=theta)
    else:
        raise ValueError('Only support gaussian,exponential,gamma distribution.')
    return funct

def function(name):
    # usage: y = function(name)(x)
    if name=='linear':
        funct = lambda x:x
    elif name=='sqrt':
        funct = lambda x: np.sign(x)*np.sqrt(abs(x)/2)
    elif name=='sin':
        funct = lambda x:np.sin(np.pi/2*x)
    elif name=='tanh':
        funct = lambda x:np.tanh(x)
    else:
        raise ValueError('Only support linear,sqrt,sin,tanh function.')
    return lambda x:funct(x)

def rand_steq(sdag,ftdag,funcs=['linear','sqrt','sin','tanh'],dists=['gaussian','gamma','exponential']):
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

        # if there are {}->node with linear function, then we do not use Gaussian distribution to ensure faithfulness
        has_linear_function = False
        for key in ssteq.keys():
            if '->' not in key:
                continue
            start,end = key.split('->')
            if end==node and ssteq[key]=='linear':
                has_linear_function = True
                break
        if has_linear_function:
            ssteq[node] = np.random.choice([dist for dist in dists if dist!='gaussian'])
        else:
            ssteq[node] = np.random.choice(dists)
        
    ftsteq = dict()
    for edge in ftdag.edges:
        cause,effect = edge
        key = '{}->{}'.format(cause,effect)

        _cause = cause.split('_')[0]; _effect = effect.split('_')[0]
        _key = '{}->{}'.format(_cause,_effect)
        
        ftsteq[key] = ssteq[_key]
        
    return ssteq,ftsteq
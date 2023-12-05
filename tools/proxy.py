import numpy as np
import pandas as pd
from binn import quantile_bin
from causallearn.utils.cit import CIT
from scipy.stats.distributions import chi2
from numpy.linalg import matrix_rank
from estimator import margEstimate,condEstimate

def proxy_cit(npdata,indX,indY,indW,significance=0.05,levely=5,lenW2level={1:[16,10],2:[18,4],3:[30,3],4:[18,2],5:[36,2]}):
    '''
    wrapper for proxy cit
    indX and indY are int index; indW is list of int indices; npdata/pddata with (size,d)
    return: True if X->Y, False if X Y
    '''
    lenW = len(indW)
    if lenW not in lenW2level.keys():
        #print('Warning: lenW: {}, not in lenW2level keys.'.format(lenW))
        raise KeyError('lenW: {}, not in lenW2level keys.'.format(lenW))
    levelx,levelw = lenW2level[lenW]
    # discretization
    tx = quantile_bin(npdata[:,indX],levelx)
    tw = [quantile_bin(npdata[:,ind],levelw) for ind in indW]
    ty = quantile_bin(npdata[:,indY],levely)

    tdata = np.stack([tx]+tw+[ty],axis=-1)
    pddata = pd.DataFrame(tdata,columns=['X']+['W{}'.format(ind) for ind in range(lenW)]+['Y'])
    # proxy test
    tester = ProxyTest(pddata,levelx,levelw,levely)
    pvalue = tester.test()
    
    if pvalue>significance:
        return False
    else:
        return True

class ProxyTest():
    def __init__(self, pddata, levelx, levelw, levely, eps=1e-5):
        '''
        pddata is a dataframe with name X,U1,...,W1,...,Y
        for cases where W is multi-variate, we presume each W' is binned into levelw categories
        '''
        self.pddata = pddata
        self.data = self.pddata.values
        self.size = self.pddata.shape[0]
        
        self.levelx = levelx
        self.levelw = levelw
        self.levely = levely
        self.eps = eps
        self.wnames = [var for var in self.pddata.columns if 'W' in var]
        
        assert levelx == self.pddata['X'].unique().shape[0]
        assert levely == self.pddata['Y'].unique().shape[0]
        levels = 1
        for wname in self.wnames:
            levels *= self.pddata[wname].unique().shape[0]
        assert levelw**len(self.wnames) == levels
        
        
    def chisq(self,):
        '''
        chi-sq test of X \perp Y | U, p>alpha indicates independence
        '''
        uindices = list()
        for ind,name in enumerate(self.pddata.columns):
            if 'U' in name:
                uindices.append(ind)
        assert len(uindices)>0
        
        _chisq = CIT(self.data, "chisq")
        p = _chisq(0,-1,uindices)
        return p 
    
    def _est_qy(self,):
        '''
        estimate q_hat:=[pr(yj|X)] for j=1,2,...,levely-1 and sigma_hat
        '''
        # q_hat
        self.prbYgivX = condEstimate(self.pddata,nameX=['Y'],levelX=[self.levely],nameC='X',levelC=self.levelx)
        self.prbYgivX[self.prbYgivX==0] = self.eps
        self.prbYgivX[self.prbYgivX==1] -= self.levely*self.eps
        
        self.q_hat = self.prbYgivX.reshape(-1,1)[:self.levelx*(self.levely-1),:]
        assert (self.q_hat!=0).all(), 'Warning, qy contains zero element.'
        
        # sigma_hat
        prbX = margEstimate(pdData=self.pddata,name='X',level=self.levelx)
        self.px_hat = np.array(prbX)[:,None]
        
        repeat_px_hat = np.tile(self.px_hat,(self.levely-1,1))
        self.sigma_hat = np.diag((self.q_hat * (1-self.q_hat) / repeat_px_hat).squeeze())
        assert np.linalg.det(self.sigma_hat)!=0, 'Warning, the matrix sigma is not invertible.'
        
        
    def _est_Q(self,):
        '''
        estimate Q_hat:=[[pr(w1|x1),...,pr(w1|xi)],
                            ...     ...    ...
                         [pr(wk|x1),...,p_hat(wk|xi)]]_(kxi) for w=w1,w2
        '''
        self.Q_hat = condEstimate(self.pddata,nameX=self.wnames,levelX=[self.levelw]*len(self.wnames),nameC='X',levelC=self.levelx)
        self.Q_hat = self.Q_hat.reshape(self.levelw**len(self.wnames),self.levelx)
        assert matrix_rank(self.Q_hat) == self.levelw**len(self.wnames), 'Warning, the matrix Q does not have full row rank. This may be because many bins in P(W|X) has no sample. Remove line-64 in estimator.py to confirm. One may use structual equations that are smoother to avoid this bug.'
        
        self.Q0_hat = np.kron(np.eye(self.levely-1,dtype=int),self.Q_hat)

    def test(self,single_level=False):
        """
        the larger (than 0.05) p-value, the more independent (H0 holds)
        """
        self._est_qy()
        self._est_Q()
        
        I = np.eye(self.levelx*(self.levely-1))
        # Sigma^-0.5 and Sigma^-1
        sigma_halfinv = np.linalg.inv(np.sqrt(self.sigma_hat))
        sigma_inv = np.linalg.inv(self.sigma_hat)

        Omega_hat = I - sigma_halfinv@self.Q0_hat.T@np.linalg.inv(self.Q0_hat@sigma_inv@self.Q0_hat.T)@self.Q0_hat@sigma_halfinv
        Xi = Omega_hat@sigma_halfinv@self.q_hat
        T = self.size*Xi.T@Xi

        # p-value
        freedom = (self.levelx - self.levelw**len(self.wnames)) * (self.levely - 1)
        p_value = chi2.sf(T.item(),freedom)
        self.T = T
        
        return p_value
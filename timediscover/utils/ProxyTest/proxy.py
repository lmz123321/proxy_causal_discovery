import numpy as np
import pandas as pd
from causallearn.utils.cit import CIT
from scipy.stats.distributions import chi2
from numpy.linalg import matrix_rank
from timediscover.utils.ProxyTest.estimator import margEstimate,condEstimate

class ProxyTest():
    def __init__(self, pddata, levelx, levelws, levely, eps=1e-5):
        '''
        pddata is a dataframe with name X,W1,...,Wd,Y
        each variable's value should be discretized into 1,2,3,...,level
        levelws is a list with length=d, so W1,...,Wd each contains levelws[0],...,levelws[d-1] levels
        '''
        self.pddata = pddata
        self.data = self.pddata.values
        self.size = self.pddata.shape[0]
        
        self.levelx = levelx
        self.levelws = levelws
        self.levely = levely
        self.eps = eps
        self.wnames = [var for var in self.pddata.columns if 'W' in var]
        
        assert levelx == self.pddata['X'].unique().shape[0]
        assert levely == self.pddata['Y'].unique().shape[0]
        levels = 1
        for wname in self.wnames:
            levels *= self.pddata[wname].unique().shape[0]
        assert np.product(levelws) == levels
    
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
        self.Q_hat = condEstimate(self.pddata,nameX=self.wnames,levelX=self.levelws,nameC='X',levelC=self.levelx)
        self.Q_hat = self.Q_hat.reshape(np.product(self.levelws),self.levelx)
        assert matrix_rank(self.Q_hat) == np.product(self.levelws), 'Warning, the matrix P(W|X) with rank: {} does not have full row rank: {}. Please use get_linind_bins() in binn.py'.format(matrix_rank(self.Q_hat),np.product(self.levelws))
        
        self.Q0_hat = np.kron(np.eye(self.levely-1,dtype=int),self.Q_hat)

    def test(self,):
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
        freedom = (self.levelx - np.product(self.levelws)) * (self.levely - 1)
        p_value = chi2.sf(T.item(),freedom)
        self.T = T
        
        return p_value
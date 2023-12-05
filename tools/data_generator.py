# adjmatrix -> summarydag -> fulltimedag -> sampledata

import numpy as np
import pandas as pd
import networkx as nx

from norm import norm
from adjmatrix import erdoes_renyi,adj0,adj1
from graph import get_parents,adj_to_dag,sdag_to_ftdag,sdag_to_ftdag_2steps
from structual_equation import distribution,function,rand_steq

class Data:
    def __init__(self,adj,k,n,beta,alpha=0.5):
        '''
        - d: number of vertices
        - k: under sampling factor, t=0,1,...,k+1
        - n: sample size 
        - alpha: control the smoothness of structure functions; recommen value 0.5
        - beta: control source var. >> mediation var.; recommend value 1.0 for small k, sqrt(10) for large k
        '''
        self.adj = adj
        self.d = self.adj.shape[0]
        self.k = k
        self.n = n 
        self.alpha = alpha
        self.beta = beta
        # summary and fulltime dags
        self.sdag = adj_to_dag(self.adj)
        self.ftdag = sdag_to_ftdag(self.sdag,self.k)
        # random st-eqs
        self.ssteq, self.ftsteq = rand_steq(self.sdag,self.ftdag)
        self.data = dict()
        self.generate()
        
    def generate(self,):
        '''
        V_i = sum_j f_j(PA_j) + exo_i
        '''
        for node in nx.topological_sort(self.ftdag):
            # exogenous dist and parents
            var,time = node.split('_'); time = int(time)
            exodist = self.ssteq[var]
            parents = get_parents(self.ftdag,node)

            if len(parents)==0:
                assert time==0, 'Node {} at time {} has no parents.'.format(var,time)
                self.data[node] = self.beta*distribution(exodist)(size=self.n)
            else:
                exo = distribution(exodist)(size=self.n)
                for parent in parents: # if var in parent => auto lagged edge  
                    key = '{}->{}'.format(parent,node) 
                    func = self.ftsteq[key]
                    exo += function(func,self.alpha)(self.data[parent]) 
                self.data[node] = exo    
        
        pddata = dict()
        for node in self.data.keys():
            var,time = node.split('_'); time = int(time)
            if time==0 or time==self.k+1:
                pddata[node] = self.data[node]
        self.pddata = pd.DataFrame(pddata)
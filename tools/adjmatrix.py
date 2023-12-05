import numpy as np
import networkx as nx

def erdoes_renyi(d,p):
    '''
    erdoes renyi model for adjmatrix, adj[i,j]=1 => i->j, each edge ~ B(p)
    '''
    fully_connect = np.triu(np.ones((d,d)),k=1)
    rand_mask = np.random.binomial(n=1,p=p,size=d**2).reshape(d,d)
    adjmatrix = fully_connect*rand_mask
    return adjmatrix

# two test summary dags
adj0 = np.array([[0,1,0,1],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
adj1 = np.array([[0,1,0,1],[0,0,1,1],[0,0,0,1],[0,0,0,0]])
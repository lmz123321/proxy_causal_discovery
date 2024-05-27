import os
os.environ['R_HOME'] = '/home/liumingzhou/Miniconda3/envs/rlanguage/lib/R' 

import numpy as np
import networkx as nx

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
robjects.packages.importr('ggm')
_ = robjects.r('''source('./time-discovery/utils/ftdag2mag/dag2mag.r')''')
rdag2mag = robjects.globalenv['DAG.to.MAG']

def get_parents(dag, node):
    '''
    node -> list of nodes
    pa is the parent of node i.i.f. (pa,node) in edges and (node,pa) not in edges
    '''
    assert node in dag.nodes, '{} not in dag.'.format(node)
    parents = list()
    for pa in dag.nodes:
        if (pa, node) in dag.edges and (node, pa) not in dag.edges:
            parents.append(pa)
    return parents

def get_rows(rmat):
    # given a r-matrix, return row names (list of string)
    return list(robjects.r['rownames'](rmat))

def mat2mag(rmagmat):
    # convert r-mag-adjmat to MAG object (nx.DiGraph with directed and bidirected edges)
    # 0->1; 10-10; 100<->100
    
    nodes = get_rows(rmagmat)
    rmagmat = np.array(rmagmat)
    mag = nx.DiGraph()
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if rmagmat[i,j]==0 and rmagmat[j,i]==1:
                mag.add_edge(nodes[j],nodes[i])
            if rmagmat[i,j]==1 and rmagmat[j,i]==0:
                mag.add_edge(nodes[i],nodes[j])
            if rmagmat[i,j]==10 and rmagmat[j,i]==10:
                raise NotImplementError('Temporal MAG without selection error does not contain - edge')
            if rmagmat[i,j]==100 and rmagmat[j,i]==100:
                mag.add_edge(nodes[i],nodes[j])
                mag.add_edge(nodes[j],nodes[i])
    return mag

def ftdag_to_mag(ftdag,k):
    '''
    ftdag (nx.DiGraph) -> ground-truth mag (nx.DiGraph with directed and bidirected edges)
    '''
    # dag->rdag (in r-ggm format)
    rdag = list()
    for node in ftdag.nodes:
        pa = get_parents(ftdag,node)
        if len(pa)>0:
            strpa = '+'.join(pa)
            formulae = '{}~{}'.format(node,strpa)
            rdag.append(robjects.Formula(formulae))
    rdagmat = robjects.r['DAG'](*rdag)
    
    # observed variable at t=0,k+1, latent variables at t=1,2,...,k
    latent = list()
    for node in ftdag.nodes:
        _,time = node.split('_')
        if int(time)!=0 and int(time)!=k+1:
            latent.append(node)
    rlatent = robjects.StrVector(set(latent))
    rselection = robjects.r('NULL')
    
    rmagmat = rdag2mag(rdagmat,rlatent,rselection)
    mag = mat2mag(rmagmat)
    return mag
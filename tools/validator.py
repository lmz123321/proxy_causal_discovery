import networkx as nx
from graph import get_parents,get_ancestors

def ftdag_to_mag(ftdag,d,k):
    '''
    from the ground-truth ftdag, return the gt-mag (directed-edges and bidirected-edges)
    '''
    # directed edges
    causes = ['X{}_0'.format(ind+1) for ind in range(d)]
    effects = ['X{}_{}'.format(ind+1,k+1) for ind in range(d)]
    diedges = list()
    for cause in causes:
        for effect in effects:
            if cause in get_ancestors(ftdag,effect):
                diedges.append((cause,effect))
    node1s = effects
    node2s = effects
    biedges = list()
    for node1 in node1s:
        for node2 in node2s:
            if node1==node2:
                continue
            anc1 = get_ancestors(ftdag,node1)
            anc2 = get_ancestors(ftdag,node2)
            intersaction = set(anc1)&set(anc2)
            # whether a *latent* confounder
            flag = False 
            for _node in intersaction:
                _,time = _node.split('_'); time = int(time)
                if time>0:
                    flag = True
                    break
            if flag and {node1,node2} not in biedges:
                biedges.append({node1,node2})
    return diedges,biedges
    
import numpy as np
import networkx as nx

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

def _get_ancestors(dag,node,ancestors,first_level=False):
    '''
    recursive approach to collect all ancestors
    call by _get_ancestors(dag,node,[],True)
    '''
    assert node in dag.nodes, '{} not in dag.'.format(node)
    if node in ancestors:
        return
    
    if not first_level:
        ancestors.append(node)
    
    parents = get_parents(dag,node)
    
    if parents:
        for parent in parents:
            _get_ancestors(dag,parent,ancestors)
            
def get_ancestors(dag,node):
    '''
    wrapper for ancestor finding
    '''
    ancestors = list()
    _get_ancestors(dag,node,ancestors,True)
    return ancestors

def get_children(dag,node):
    '''
    node -> list of nodes
    ch is the child of node i.i.f. (node,ch) is in edges and (ch,node) not in edges
    '''
    assert node in dag.nodes,'{} not in dag'.format(node)
    children = list()
    for ch in dag.nodes:
        if (node,ch) in dag.edges and (ch,node) not in dag.edges:
            children.append(ch)
    return children

def _get_descendants(dag,node,descendants,first_level=False):
    '''
    recursive approach to collect all descendants
    call by _get_descendants(dag,node,[],True)
    '''
    assert node in dag.nodes,'{} not in dag.'.format(node)
    if node in descendants:
        return
    
    if not first_level:
        descendants.append(node)
        
    children = get_children(dag,node)
    
    if children:
        for child in children:
            _get_descendants(dag,child,descendants)
            
def get_descendants(dag,node):
    '''
    wrapper for descendant finding
    '''
    descendants = list()
    _get_descendants(dag,node,descendants,True)
    return descendants

def len_path(path):
    '''
    the length of a path [start,V1,...,Vn,end] is defined as n
    '''
    return len(path)-2

def is_direct(dag,path):
    '''
    whether the path is a direced one in the dag; we defaultly also check whether the path is a connected one
    '''
    for start,end in [path[i:i+2] for i in range(len(path) - 1)]:
        if (start,end) in dag.edges:
            continue
        elif (end,start) in dag.edges:
            return False
        else:
            raise ValueError('{}-{} not in dag.'.format(start,end))
    return True

def all_direct_paths(dag,start,end,maxlen):
    '''
    return all directed path from start to end with 0<len<=maxlen
    '''
    assert (start,end) in dag.edges, '{}->{} not in dag'.format(start,end)
    shortpaths = list()
    longpaths = list()
    for path in nx.all_simple_paths(dag, source=start, target=end, cutoff=maxlen+1):
        if is_direct(dag,path):
            length = len_path(path)
            if 0<length and length<=maxlen-1:
                shortpaths.append(path)
            elif length==maxlen:
                longpaths.append(path)
                
    return shortpaths,longpaths

def all_direct_confound_paths(dag,start,end,maxlen):
    '''
    return all directed path from start to end with 0<=len<=maxlen
    this is for confounding structure detection
    '''
    paths = list()
    for path in nx.all_simple_paths(dag, source=start, target=end, cutoff=maxlen+1):
        if is_direct(dag,path):
            paths.append(path)
    return paths

def all_confound_structure(dag,start,end,maxr,maxq):
    '''
    return a list, containing all (r,q)-confounding structure between start and end
    0<=r<=maxr; 0<=q<=maxq; the default value for maxr/maxq should be k-1
    '''
    common_ancs = set(get_ancestors(dag,start)) & set(get_ancestors(dag,end))
    confounds = list()
    for anc in common_ancs:
        # directed path from common anc to start/end
        to_start = all_direct_confound_paths(dag,anc,start,maxr)
        _to_end = all_direct_confound_paths(dag,anc,end,maxq)

        # start should not in any to_end path
        to_end = list()
        for path in _to_end:
            if start not in path:
                to_end.append(path)
        if len(to_end)>0 and len(to_start)>0:
            confound = {'anc':anc,'to_start':to_start,'to_end':to_end}
            confounds.append(confound)
    return confounds

def get_Q(mag,pcdag,start,end,d,k):
    '''
    detect the set Q from mag (tuple of (diedges,biedges))
    for start-->end, Q contains start and any Qi!=end s.t.
    1. start(t)->Qi(t+k+1) in the mag
    2. Qi is not end's descentant
    '''
    assert (start,end,{'style':'-->'}) in pcdag.edges.data(), '{}-->{} not in pcdag.'.format(start,end)
    Q = [start]
    for diedge in mag[0]:
        cause,effect = diedge
        cause,effect = cause.split('_')[0], effect.split('_')[0]
        # for cause->effect in MAG, effect in Q if:
        # 1. cause is start
        # 2. effect is not end
        # 3. effect not in Dec(end)
        flag1 = cause==start
        flag2 = effect!=end
        flag3 = effect not in get_descendants(pcdag,end)

        if flag1 and flag2 and flag3:
            Q.append(effect)
            
    # remove the redundance on 'start
    Q = list(set(Q))
    return Q

def is_certain(pcdag):
    '''
    judge whether the pcdag is fully assured; i.e. all --> edges are removed or turned to ->
    '''
    for edge in pcdag.edges.data():
        style = edge[-1]['style']
        if style=='-->':
            return False
    return True

def adj_to_dag(adj):
    '''
    adj matrix -> nx.DiGraph, X1,...,Xd
    '''
    d = adj.shape[0]
    try:
        dag = nx.from_numpy_matrix(adj,create_using=nx.DiGraph)
    except AttributeError:
        dag = nx.DiGraph(np.array(adj)) # for networkx>3.0
        
    mapping = dict(zip(range(d),['X{}'.format(ind+1) for ind in range(d)]))
    dag = nx.relabel_nodes(dag, mapping)
    return dag

def sdag_to_ftdag(sdag,k):
    '''
    summary dag -> full time dag; k the under sampling factor, t=0,1,...,k+1
    '''
    ftdag = nx.DiGraph()
    
    # add nodes
    nodes = list()
    for _node in sdag.nodes:
        nodes += ['{}_{}'.format(_node,t) for t in range(k+2)]
    ftdag.add_nodes_from(nodes)
    
    # add edges
    for _node in ftdag.nodes:
        # variable name and time step
        var,time = _node.split('_'); time = int(time)
        if time==0:
            continue
        # parents in summary dag
        spa = get_parents(sdag,var)
        # parents in fulltime dag
        ftpa = ['{}_{}'.format(var,time-1)]
        for _spa in spa:
            ftpa.append('{}_{}'.format(_spa,time-1))

        for _ftpa in ftpa:
            ftdag.add_edge(_ftpa,_node)    

    return ftdag


def sdag_to_ftdag_2steps(sdag,k):
    '''
    summary dag -> full time dag; k the under sampling factor, t=0,1,...,2k+2
    '''
    ftdag = nx.DiGraph()
    
    # add nodes
    nodes = list()
    for _node in sdag.nodes:
        nodes += ['{}_{}'.format(_node,t) for t in range(2*k+3)]
    ftdag.add_nodes_from(nodes)
    
    # add edges
    for _node in ftdag.nodes:
        # variable name and time step
        var,time = _node.split('_'); time = int(time)
        if time==0:
            continue
        # parents in summary dag
        spa = get_parents(sdag,var)
        # parents in fulltime dag
        ftpa = ['{}_{}'.format(var,time-1)]
        for _spa in spa:
            ftpa.append('{}_{}'.format(_spa,time-1))

        for _ftpa in ftpa:
            ftdag.add_edge(_ftpa,_node)    

    return ftdag
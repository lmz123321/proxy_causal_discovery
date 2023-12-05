from tqdm import tqdm
import networkx as nx
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.timeFCI import timefci
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from graph import get_ancestors,all_direct_paths,all_confound_structure

def fci_time(dataset,citest,node_names,significance=0.05,need_bk=True,real_world=False,cache_path=None, **kwargs):
    '''
    fci over {X(t0)} and {X(t_k+1)}, the background knowledge includes:
    1.1 all lags are directed
    1.2 all instants are bi-directed
    2. auto-lags always exist
    '''
    nodes = []
    for ind in range(dataset.shape[1]):
        node = GraphNode(node_names[ind]); node.add_attribute('id',ind); nodes.append(node)
    
    maxtime = 0
    for name in node_names:
        _,t = name.split('_'); t=int(t)
        if t>maxtime:
            maxtime = t
            
    # background knowledges
    patternTonode = dict()
    for node in nodes:
        patternTonode[node.name] = node
        
    # background-1: edge from larger tier to smaller one is forbidden; edge between the same tier are bi-directed
    patternTotier = dict()
    for node in node_names:
        var,time = node.split('_'); time = int(time)
        patternTotier[node] = time

    bk = BackgroundKnowledge()
    for pattern in patternTotier.keys():
        bk.add_node_to_tier(patternTonode[pattern], patternTotier[pattern])
        
    # background-2: auto-lag always exists
    for ind in range(dataset.shape[1]//2):
        cause = 'X{}_{}'.format(ind+1,0)
        effect = 'X{}_{}'.format(ind+1,maxtime)
        bk.add_required_by_node(patternTonode[cause], patternTonode[effect])

    if not need_bk:
        bk = None

    method = timefci if real_world else fci
    G, edges = method(dataset=dataset, nodes=nodes, independence_test_method=citest, alpha=significance,
               background_knowledge=bk, cache_path=cache_path, **kwargs)
    
    return edges

def post_process(edges,real_world=False):
    '''
    list of Edge object, to directed edges and bidirected edges
    for real-world dataset, bi-directed edge is the union of that in t,t+k+1
    '''
    diedges = list(); biedges = list()
    for edge in edges:
        cause = edge.get_node1().name
        effect = edge.get_node2().name
        type1 = edge.get_endpoint1().value
        type2 = edge.get_endpoint2().value
        if type1==-1 and type2==1:
            diedges.append((cause,effect))
        elif type1==1 and type2==1:
            _,t1 = cause.split('_'); t1 = int(t1)
            _,t2 = effect.split('_'); t2 = int(t2)
            
            if not real_world:
                if t1==0 or t2==0:
                    continue
            biedges.append({cause,effect})
        else:
            raise ValueError('Expect (-1,1) or (1,1), got ({},{})'.format(type1,type2))
    return diedges,biedges

def mag_to_pcdag(edges,d,k):
    '''
    FCI-MAG to Partially Connected DAG
    - edge type: (-1,1) ->; (1,1) <->
    - d: node number
    - k: under sample factor
    '''
    # directed-edges: list of tuples, bidirected-edges: list of sets
    if type(edges)==tuple:
        diedges,biedges = edges
    else:
        diedges,biedges = post_process(edges)
            
    pcdag = nx.DiGraph()
    pcdag.add_nodes_from(['X{}'.format(ind+1) for ind in range(d)])
    for cause in pcdag.nodes:
        for effect in pcdag.nodes:
            flag1 = ('{}_0'.format(cause),'{}_{}'.format(effect,k+1)) in diedges
            
            flag2 = {'{}_{}'.format(cause,0),'{}_{}'.format(effect,0)} in biedges
            flag3 = {'{}_{}'.format(cause,k+1),'{}_{}'.format(effect,k+1)} in biedges
            if flag1 and (flag2 or flag3):
                pcdag.add_edge(cause,effect,style='-->')
    return pcdag


def rulea_(pcdag,k):
    '''
    given a pcdag constructed from mag with all edges -->; use rule-a to turn some edges to ->
    note: this is an in-place function
    '''
    for edge in pcdag.edges.data():
        start,end,style = edge; style=style['style']
        #assert style=='-->','PC-DAG has {}->{} before rule-a.'.format(start,end)

        shorts,longs = all_direct_paths(pcdag,start,end,k)
        confounds = all_confound_structure(pcdag,start,end,k-1,k-1)

        rulea = len(shorts)>0 or (len(longs)>0 and len(confounds)>0)
        if not rulea:
            edge[2]['style'] = '->'
    return pcdag
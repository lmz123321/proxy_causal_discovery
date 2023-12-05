import networkx as nx
import matplotlib.pyplot as plt

def draw_ftdag(dag):
    '''
    draw full time dag
    '''
    nodes = dag.nodes
    
    # variable names and max time step
    vares = set()
    time = 0
    for node in nodes:
        var,t = node.split('_'); t = int(t)
        if t>time:
            time = t
        vares.add(var)
    vares = sorted(list(vares))
    d = len(vares)
    
    # relative positions
    y = range(d); x = range(time+1)
    poses = dict()
    for ynd, var in enumerate(vares):
        for xnd,t in enumerate(range(time+1)):
            node = '{}_{}'.format(var,t)
            pos = [x[xnd],y[ynd]]
            poses[node] = pos
            
    plt.figure()
    nx.draw(dag, pos=poses,with_labels=True,
        edgecolors='black',node_size=1200,node_color='aliceblue',width=1.5,linewidths=1.5,font_weight='normal')
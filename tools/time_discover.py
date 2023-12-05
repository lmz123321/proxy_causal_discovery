from algorithm import fci_time,post_process,mag_to_pcdag,rulea_
from graph import is_certain,get_Q
from proxy import proxy_cit
import networkx as nx
from metric import precision,recall,f1
from validator import ftdag_to_mag
from jsonio import _save,_load
from tqdm import tqdm
import os
import numpy as np

def time_discover(dataset,node_names,d,k,citest,real_world,cache_path=None,significance=0.05,
                  debug=False,sdag=None,ftdag=None,**kwargs):
    '''
    functs: - recover sdag from data at t0,t_k+1
    params: - dataset with shape (size,d); node_names a list of len=d
            - for debug mode, will record mediation precision/recall, requres gt-sdag/ftdag input
            - set real_world=False in simulation data, will assume Xi(t=0) are all independent
    logic: 
            - fci_time(data.t0,t_k+1) -> mag
            - mag_to_pcdag(mag) -> pcdag
            - while(not is_certain(pcdag))
               - rulea_(pcdag)
               # rule-b
               - for --> edge in pcdag:
               -     detect_Q(edge)
               - for the --> edge with minimal Q
               -     proxt_cit: delete or set to ->
    '''
    if cache_path is not None:
        try:
            os.makedirs(cache_path)
        except OSError:
            print('Warning: {} exists, we recommend using an empty folder as cache.'.format(cache_path))
            os.makedirs(cache_path,exist_ok=True)

    if debug: # debug mode
        info = dict()
    
    # mag
    mag = fci_time(dataset,citest,node_names,significance,real_world=real_world,cache_path=cache_path,**kwargs)
    diedges,biedges = post_process(mag,real_world)

    if cache_path is not None:
        _save(diedges,os.path.join(cache_path,'mag_diedges.json'))
        _save([tuple(e) for e in biedges],os.path.join(cache_path,'mag_biedges.json'))
    if debug:
        digts, bigts = ftdag_to_mag(ftdag,d,k)
        #info['mag_di'] = [precision(diedges,digts),recall(diedges,digts)]
        #info['mag_bi'] = [precision(biedges,bigts),recall(biedges,bigts)]
        mag_pred = diedges + biedges
        mag_gt = digts + bigts
        info['mag'] = [f1(mag_pred,mag_gt),precision(mag_pred,mag_gt),recall(mag_pred,mag_gt)]
        
    # pd-dag
    pcdag = mag_to_pcdag((diedges,biedges),d,k)
    if cache_path is not None:
        nx.write_gml(pcdag,os.path.join(cache_path,'pcdag.gml'))
    try:
        dicycles = nx.find_cycle(pcdag,orientation='original')
        #print('Warning: directed cycles detected in pc-dag',dicycles)
    except:
        pass 
            
    if debug:
        gt_pcdag = mag_to_pcdag((digts,bigts),d,k)
        info['pcdag'] = [f1(pcdag.edges,gt_pcdag.edges),
                         precision(pcdag.edges,gt_pcdag.edges),recall(pcdag.edges,gt_pcdag.edges)]
        assert None not in info['pcdag']

    # proxy cit
    #info['proxy'] = list() # [(pred,gt),(pred,gt),...] for each proxy call

    #pbar = tqdm(total=len(pcdag.edges))
    #pbar.set_description('Verify {} edges in pcdag'.format(len(pcdag.edges)))

    while not is_certain(pcdag):
        #pbar.update()
        # rule-a
        pcdag = rulea_(pcdag,k)
        if is_certain(pcdag):
            break
        # rule-b
        minQsize = d; minQedge = None; minQ = None
        for edge in pcdag.edges.data():
            start,end,style = edge; style=style['style']
            if style=='->':
                continue
            Q = get_Q((diedges,biedges),pcdag,start,end,d,k)
            if len(Q)<minQsize:
                minQsize = len(Q)
                minQedge = edge
                minQ = Q
        # decide keep or remove the minQedge via proxy cit
        start,end,_ = minQedge; node_names = list(node_names)
        indX = node_names.index('{}_0'.format(start))
        indY = node_names.index('{}_{}'.format(end,k+1))
        indW = [node_names.index('{}_{}'.format(nameQ,k+1)) for nameQ in minQ]

        try:
            has_edge = proxy_cit(dataset, indX, indY, indW, significance)
        except:
            # todo: find a more elegant way to solve situations when proxy test fails,
            has_edge = True
            print('Warning: unknown proxy test error (highly likely levelw>5), set the edge by default.')
           
        if has_edge:
            minQedge[-1]['style'] = '->'
        else:
            pcdag.remove_edge(start,end)

        if debug:
            gt_edge = (start,end) in sdag.edges
            #info['proxy'].append([has_edge,gt_edge])
    #pbar.close()

    if debug:
        return pcdag,info
    else:
        return pcdag
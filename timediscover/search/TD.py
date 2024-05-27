import networkx as nx
from timediscover.graph.NxGraph import detect_dicycle_in_mag,mag_to_pddag,is_certain,all_confound_structure,all_direct_paths,search_set_M,get_set_M,get_set_S
from timediscover.search.PC import pc_time,pc_fdr_time
from timediscover.utils.BY_procedure import Benjamini_Yekutieli
from timediscover.utils.cit import CIT
from timediscover.utils.ProxyTest.proxy_test import ProxyCITest

from typing import List
from numpy import ndarray
from timediscover.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

def td(
        data: ndarray,
        indep_test=str,
        proxy_test=str,
        node_names: List[str] | None = None,
        subsampling_factor=int,
        alpha: float = 0.05,
        stable: bool = True,
        background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = False,
        proxy_ratio=1,
        **kwargs):
    """
    Implement the Time-Discovery algorithm
    Args: data: observation data, nd.array with shape (n,d), n-sample size, d-node numbers
          indep_test: method for CI Test, choose from [info_test, kci, d_separation, fisherz, ...]
                      if choose d_separation, please provide the gt-ftdag by specifying "true_dag = data_gen.ftdag"
          proxy_test: method for Proxy Test, choose from [proxy_test, d_separation]
          node_names: nodes names of data, e.g., [X1_0,X1_5,X2_0,X2_5...]
          subsampling_factor: k-1, e.g. sub_factor=1 means A(0), A(2), A(4) can be observed
          alpha (default 0.1): level of FDR
          stable: whether use stable-PC
          proxy_ratio (default 1.0): proxy-test conditioning on the observed confounders

          ProxyTest-related args, levelx,levelw,levely,ratio, should be specified in **kwargs
    Example:
          adj = erdoes_renyi(5,0.3)
          data_gen = TimeData(adj,k=1,n=200)
          mag,sdag = td(data = data_gen.observation.values,node_names = data_gen.observation.columns,
                        indep_test = 'd_separation', # set to 'fihserz','kci','info_test', ... if you don't want Oracle test
                        proxy_test = 'd_separation', # set to 'proxy_test' if you don't want Oracle test
                        alpha = 0.05, subsampling_factor=data_gen.k,
                        true_dag = data_gen.ftdag # if you need Oracle CI/Proxy Test)
          # Evaluation
          prec,rec = precision_skeleton(mag,data_gen.mag_ske),recall_skeleton(mag,data_gen.mag_ske)
          print('Precision: {}, Recall: {} of MAG'.format(prec,rec))
          prec,rec = precision(sdag.edges,data_gen.sdag.edges), recall(sdag.edges,data_gen.sdag.edges)
          print('Precision: {}, Recall: {} of sDAG'.format(prec,rec))
    """
    # 1. recover MAG
    kwargs['oracle_citest_node_names'] = node_names
    mag = pc_time(data, alpha=alpha, node_names=node_names, stable=stable,
                  background_knowledge=background_knowledge, indep_test=indep_test,
                  show_progress=show_progress, **kwargs)
    # detect_dicycle_in_mag(mag)
    mag_ske = mag[0] + mag[1]

    # 2. construct PD-DAG from MAG
    static_node_names = list()
    for name in [name.split('_')[0] for name in node_names]:
        if name not in static_node_names:
            static_node_names.append(name)
    pddag = mag_to_pddag(mag, subsampling_factor, static_node_names)

    # 3. initiate proxy searching (step-3 of alg.1)
    proxytester = ProxyCITest(data, node_names)
    while not is_certain(pddag):
        # rule-a
        pddag = rulea(pddag, subsampling_factor)
        if is_certain(pddag):
            break
        # rule-b
        pddag = ruleb(pddag, mag, proxy_test, proxytester, proxy_ratio, alpha, **kwargs)

    return mag_ske, pddag


def rulea(pddag, k):
    '''
    Given a pddag constructed from mag with all edges -->; use rule-a to turn some edges to ->
    '''
    for edge in pddag.edges.data():
        start, end, style = edge;
        style = style['style']
        shorts, longs = all_direct_paths(pddag, start, end, k)
        confounds = all_confound_structure(pddag, start, end, k - 1, k - 1)

        rulea = len(shorts) > 0 or (len(longs) > 0 and len(confounds) > 0)
        if not rulea:
            edge[2]['style'] = '->'
    return pddag


def ruleb(pddag, mag, proxy_test, proxytester, proxy_ratio, alpha, **kwargs):
    '''
    Given a pddag processed with rule-a; use rule-b to turn one edge to -> or delete it
    '''
    # search for --> with minimal Mset
    minMsize = len(pddag.nodes)
    minEdge, minMset, minSset = None, None, None

    for edge in pddag.edges.data():
        start, end, style = edge;
        style = style['style']
        if style == '->':
            continue
        Mset = search_set_M(mag, pddag.nodes, start, end)
        Sset = get_set_S(mag, pddag.nodes, Mset, start, end)
        if len(Mset) < minMsize:
            minMsize = len(Mset)
            minEdge = edge
            minMset = Mset
            minSset = Sset

    # proxy test whether remove or assure this edge
    A, B, _ = minEdge
    k = mag[0][0][1].split('_')[1]

    if proxy_test == 'd_separation':
        minMset = {M + '_1' for M in minMset}
        pvalue = nx.d_separated(kwargs['true_dag'], {A + '_0'}, {B + '_' + k}, minMset.union(minSset))
    else:
        minMset = {M + '_' + k for M in minMset}
        pvalue = proxytester(X=A + '_0', Y=B + '_' + k, W=list(minMset), C=list(minSset), ratio=proxy_ratio)

    if pvalue > alpha:
        pddag.remove_edge(A, B)
    else:
        minEdge[-1]['style'] = '->'  # this is an inplace operation
    return pddag


def td_fdr(
        data: ndarray,
        indep_test=str,
        proxy_test=str,
        node_names: List[str] | None = None,
        alpha: float = 0.1,
        stable: bool = True,
        background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = False,
        proxy_ratio = 1,  
        **kwargs):
    """
    Implement the Time-Discovery algorithm with FDR control
    Args: data: observation data, nd.array with shape (n,d), n-sample size, d-node numbers
          indep_test: method for CI Test, choose from [info_test, kci, d_separation, fisherz, ...]
                      if choose d_separation, please provide the gt-ftdag by specifying "true_dag = data_gen.ftdag"
          proxy_test: method for Proxy Test, choose from [proxy_test, d_separation]
          node_names: nodes names of data, e.g., [X1_0,X1_5,X2_0,X2_5...]
          alpha (default 0.1): level of FDR
          stable: whether use stable-PC
          proxy_ratio (default 1.0): proxy-test conditioning on the observed confounders

          ProxyTest-related args, levelx,levelw,levely,ratio, should be specified in **kwargs
    Example:
          adj = erdoes_renyi(5,0.3)
          data_gen = TimeData(adj,3,600)
          mag,sdag = td(data = data_gen.observation.values,
                        indep_test = 'info_test'
                        proxy_test = 'proxy_test'
                        node_names = data_gen.observation.columns
                        alpha = 0.05,
                        true_dag = data_gen.ftdag) # if you need Oracle CI/Proxy Test)
          # Evaluation
          prec,rec = precision_skeleton(mag,data_gen.mag_ske),recall_skeleton(mag,data_gen.mag_ske)
          print('Precision: {}, Recall: {} of MAG'.format(prec,rec))
          prec,rec = precision(sdag.edges,data_gen.sdag.edges), recall(sdag.edges,data_gen.sdag.edges)
          print('Precision: {}, Recall: {} of sDAG'.format(prec,rec))
    """
    kwargs['oracle_citest_node_names'] = node_names

    if 'true_mag' in kwargs.keys():
        mag = kwargs['true_mag'] # you can also specify indep_test='d_separation' and run the pd_fdr_time() to obtain the mag.
        # we tried and found the results are exactly identical. so here we directly used the gt-mag to save time.
    else:
        mag = pc_fdr_time(data, alpha=alpha,
                          node_names=node_names, stable=stable, background_knowledge=background_knowledge,
                          indep_test=indep_test,
                          show_progress=show_progress, **kwargs)

    detect_dicycle_in_mag(mag)
    mag_ske = mag[0] + mag[1]

    static_node_names = list()
    for name in [name.split('_')[0] for name in node_names]:
        if name not in static_node_names:
            static_node_names.append(name)

    proxytest = ProxyCITest(data, node_names)
    pvalues = list()
    k = mag[0][0][1].split('_')[1]

    for A in static_node_names:
        for B in static_node_names:
            if A != B:
                flag1 = (A + '_0', B + '_' + k) in mag[0]
                flag2 = (A + '_' + k, B + '_' + k) in mag[1]
                flag3 = (B + '_' + k, A + '_' + k) in mag[1]
                if flag1 and (flag2 or flag3):
                    Mset = get_set_M(mag, static_node_names, A, B)
                    Sset = get_set_S(mag, static_node_names, Mset, A, B)

                    # At ind Bt+k | Mt+1 cup St
                    if proxy_test == 'd_separation':
                        Mset = {M + '_1' for M in Mset}
                        pvalue = nx.d_separated(kwargs['true_dag'], {A + '_0'}, {B + '_' + k}, Mset.union(Sset))
                    else:
                        Mset = {M + '_' + k for M in Mset}
                        pvalue = proxytest(X=A + '_0', Y=B + '_' + k, W=list(Mset), C=list(Sset), ratio=proxy_ratio)
                    pvalues.append(((A, B), pvalue))

    reject_id, _ = Benjamini_Yekutieli(pvalues, alpha)
    sdag = nx.DiGraph()
    for edge in reject_id:
        sdag.add_edge(edge[0], edge[1])
    return mag_ske, sdag
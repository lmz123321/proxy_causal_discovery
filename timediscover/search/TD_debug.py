import networkx as nx

from timediscover.graph.NxGraph import detect_dicycle_in_mag,get_set_M,get_set_S
from timediscover.search.PC import pc_fdr_time
from timediscover.utils.BY_procedure import Benjamini_Yekutieli
from timediscover.utils.cit import CIT
from timediscover.utils.ProxyTest.proxy_test import ProxyCITest

from typing import List
from numpy import ndarray
from timediscover.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

# for DEBUG purpose only
# we want to check the averaged ratio of gt=H0 in our BY-procedures calls

def td(
        data: ndarray,
        indep_test=str,
        proxy_test=str,
        node_names: List[str] | None = None,
        alpha: float = 0.7,
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
          alpha (default 0.7): level of FDR
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

    static_node_names = list()
    for name in [name.split('_')[0] for name in node_names]:
        if name not in static_node_names:
            static_node_names.append(name)

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
                    Mset = {M + '_1' for M in Mset}
                    pvalue = nx.d_separated(kwargs['true_dag'], {A + '_0'}, {B + '_' + k}, Mset.union(Sset))
                    pvalues.append(pvalue)
    return pvalues
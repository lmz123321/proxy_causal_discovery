from __future__ import annotations

import time
import warnings
from itertools import combinations, permutations
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
from numpy import ndarray

from timediscover.graph.GraphClass import CausalGraph
from timediscover.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from timediscover.utils.cit import CIT
from timediscover.search import SkeletonDiscovery, SkeletonDiscovery_FDR,SkeletonDiscovery_FDR_Time,SkeletonDiscovery_Time


def pc(
    data: ndarray, 
    alpha = 0.05,
    indep_test= str,
    stable: bool = True, 
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False, 
    show_progress: bool = True,
    node_names: List[str] | None = None,
    **kwargs
):
    """
    Implement PC_skeleton algorithm
    Args: data, a numpy array with shape (n,d), n is sample size, d is vertex number
          alpha, significance level
          indep_test, string, choose from ['info_test','fisherz','kci','d_separation',...]
    """
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    ind_test = CIT(data, indep_test, **kwargs)
    cg = SkeletonDiscovery.skeleton_discovery(data, alpha, ind_test, stable,
                                              background_knowledge=background_knowledge, verbose=verbose,
                                              show_progress=show_progress, node_names=node_names,**kwargs)

    return cg

def pc_fdr(
    data: ndarray,
    alpha = 0.05,
    indep_test= str,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None,
    show_progress: bool = True,
    node_names: List[str] | None = None,
    **kwargs
):
    """
    Implement PC_skeleton_fdr algorithm
    Args: data, a numpy array with shape (n,d), n is sample size, d is vertex number
          alpha, desired FDR level
          indep_test, string, choose from ['info_test','fisherz','kci','d_separation',...]
    """
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    ind_test = CIT(data, indep_test, **kwargs)
    cg = SkeletonDiscovery_FDR.skeleton_discovery_fdr(data, alpha, ind_test, stable,
                                              background_knowledge=background_knowledge,
                                              show_progress=show_progress, node_names=node_names, **kwargs)

    return cg


def pc_fdr_time(
        data: ndarray,
        alpha=0.05,
        indep_test=str,
        stable: bool = True,
        background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = True,
        node_names: List[str] | None = None,
        **kwargs
):
    """
    Implement PC_skeleton_fdr_time algorithm
    Args: data, a numpy array with shape (n,d), n is sample size, d is vertex number
          alpha, desired FDR level
          indep_test, string, choose from ['info_test','fisherz','kci','d_separation',...]
    Return:
         mag = [list_of_direct_edges, list_of_bidirected_edges]
         list_of_direct_edges = [(X1_0,X3_3), (X4_0,X5_3), ...] each tuple is always directed
    """
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    msg = 'The PC_fdr_time algorithm should be used very carefully. \n\n' + \
          'We use the following rules to decide sepset(a,b): \n' + \
          '1. time(a)=time(b)=0: delete the edge without testing (for simulation, there is always no edge; for real-world, we dont have data at -k-1 and any detected edge can be unreliable) \n' + \
          '2. time(a)=0,time(b)=k+1 (and vice versa): search sepset in time=0 (actually unnecessary for simulation data) \n' + \
          '3. time(a)=time(b)=k+1: search sepset in time=0 \n'
    # warnings.warn(msg)

    ind_test = CIT(data, indep_test, **kwargs)
    cg = SkeletonDiscovery_FDR_Time.skeleton_discovery_fdr_time(data, alpha, ind_test, stable,
                                                      background_knowledge=background_knowledge,
                                                      show_progress=show_progress, node_names=node_names, **kwargs)


    mag = [list(), list()]  # directed edges and bidirected edges
    d = len(cg.G.nodes)
    for i in range(d):
        for j in range(d):
            if cg.G.graph[i, j] != 0 and cg.G.graph[j, i] != 0:
                name_i, time_i = cg.G.nodes[i].name, int(cg.G.nodes[i].name.split('_')[1])
                name_j, time_j = cg.G.nodes[j].name, int(cg.G.nodes[j].name.split('_')[1])
                if time_i < time_j:
                    mag[0].append((name_i, name_j))
                if time_i == time_j and ((name_j, name_i) not in mag[0] + mag[1]):
                    mag[1].append((name_i, name_j))

    return mag


def pc_time(
        data: ndarray,
        alpha=0.05,
        indep_test=str,
        stable: bool = True,
        background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = True,
        node_names: List[str] | None = None,
        **kwargs
):
    """
    Implement PC_skeleton_time algorithm
    Args: data, a numpy array with shape (n,d), n is sample size, d is vertex number
          alpha, desired FDR level
          indep_test, string, choose from ['info_test','fisherz','kci','d_separation',...]
    Return:
         mag = [list_of_direct_edges, list_of_bidirected_edges]
         list_of_direct_edges = [(X1_0,X3_3), (X4_0,X5_3), ...] each tuple is always directed
    """
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    msg = 'The PC_time algorithm should be used very carefully. \n\n' + \
          'We use the following rules to decide sepset(a,b): \n' + \
          '1. time(a)=time(b)=0: delete the edge without testing (for simulation, there is always no edge; for real-world, we dont have data at -k-1 and any detected edge can be unreliable) \n' + \
          '2. time(a)=0,time(b)=k+1 (and vice versa): search sepset in time=0 (actually unnecessary for simulation data) \n' + \
          '3. time(a)=time(b)=k+1: search sepset in time=0 \n'
    # warnings.warn(msg)

    ind_test = CIT(data, indep_test, **kwargs)
    cg = SkeletonDiscovery_Time.skeleton_discovery_time(data, alpha, ind_test, stable,
                                                      background_knowledge=background_knowledge,
                                                      show_progress=show_progress,node_names=node_names)


    mag = [list(), list()]  # directed edges and bidirected edges
    d = len(cg.G.nodes)
    for i in range(d):
        for j in range(d):
            if cg.G.graph[i, j] != 0 and cg.G.graph[j, i] != 0:
                name_i, time_i = cg.G.nodes[i].name, int(cg.G.nodes[i].name.split('_')[1])
                name_j, time_j = cg.G.nodes[j].name, int(cg.G.nodes[j].name.split('_')[1])
                if time_i < time_j:
                    mag[0].append((name_i, name_j))
                if time_i == time_j and ((name_j, name_i) not in mag[0] + mag[1]):
                    mag[1].append((name_i, name_j))

    return mag
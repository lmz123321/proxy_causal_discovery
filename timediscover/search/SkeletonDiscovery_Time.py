from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy import ndarray
from typing import List
from tqdm import tqdm

from timediscover.graph.GraphClass import CausalGraph
from timediscover.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from timediscover.utils.PCUtils.Helper import append_value
from timediscover.utils.cit import CIT


def skeleton_discovery_time(
    data: ndarray,
    alpha: float,
    indep_test: CIT,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None,
    maxFanIn: int = -1,
) -> CausalGraph:
    """
    Perform skeleton discovery for time-series MAG
    We use the following rules to decide sepset(a,b):
        1. time(a)=time(b)=0: delete the edge without testing (for simulation, there is always no edge;
            for real-world, we dont have data at -k-1 and any detected edge can be unreliable)
        2. time(a)=0,time(b)=k+1 (and vice versa): search sepset in time=0 (actually unnecessary for simulation data)
        3. time(a)=time(b)=k+1: search sepset in time=0
    The validity of the above rules are proved in Claim-1 in the Annals paper

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : class CIT, the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]

    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)
    # the 'depth' argument is the same as the 'n' argument in PC algorithm
    # when depth=1, mean searching for sep-set with cardinality=1, ...
    # so, we can set a threshold for max-depth, namely maxFanIn, to control the searching range and reduce comutation cost
    depth = -1
    if show_progress:
        pbar = tqdm(total=no_of_var)
    while cg.max_degree() - 1 > depth:
        depth += 1
        # custom argument 10-11
        if maxFanIn != -1 and depth>maxFanIn:
            break
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
                pbar.set_description(f'Depth={depth}, working on node {x}')

            time_x = int(node_names[x].split('_')[1])
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue

            for y in Neigh_x:
                time_y = int(node_names[y].split('_')[1])
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                if time_x == 0 and time_y == 0:
                    knowledge_ban_edge = True

                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))  # after all conditioning sets at
                        edge_removal.append((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    Stime = [int(node_names[node].split('_')[1]) for node in S]
                    if not (len(Stime) == 0 or sum(Stime) == 0):
                        continue  # only search sepset in time=0

                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            break
                        else:
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            for s in S:
                                sepsets.add(s)
                    else:
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress:
        pbar.close()

    return cg

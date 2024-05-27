from timediscover.utils.BY_procedure import Benjamini_Yekutieli
from itertools import combinations, permutations

import numpy as np
from numpy import ndarray
from typing import List
from tqdm import tqdm

from timediscover.graph.GraphClass import CausalGraph
from timediscover.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from timediscover.utils.PCUtils.Helper import append_value
from timediscover.utils.cit import CIT


def skeleton_discovery_fdr(
        data: ndarray,
        alpha: float,
        indep_test: CIT,
        stable: bool = True,
        background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = True,
        node_names: List[str] | None = None,
        maxFanIn: int = -1,
) -> CausalGraph:
    """
    Perform skeleton discovery with FDR
    Li et al. 2009 Controlling the False Discovery Rate of the Association/Causality Structure Learned with the PC Algorithm

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired FDR level in (0,1)
    indep_test : class CIT, the independence test being used
            [info_test, fisherz, chisq, gsq, mv_fisherz, kci, d_separation]
           - info_test: mutual information based ind. test
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
           - d_separation: Oracle CI test
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    show_progress : True iff the algorithm progress should be show in console.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)
    maxFanIn: maximum range to search for the cond. set (default=-1, search all possible sets)
    Returns
    -------
    cg : a CausalGraph object. Where  cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j
    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)
    depth = -1

    if show_progress:
        pbar = tqdm(total=no_of_var)

    ordered_pairs = list(permutations(range(no_of_var), 2))
    Pmax = {'({},{})'.format(x, y): -1 for (x, y) in ordered_pairs}

    while cg.max_degree() - 1 > depth:
        depth += 1
        if maxFanIn != -1 and depth > maxFanIn:
            break
        edge_removal = []
        if show_progress:
            pbar.reset()

        for x in range(no_of_var):
            if show_progress:
                pbar.update()
                pbar.set_description(f'Depth={depth}, working on node {x}')

            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue

            for y in Neigh_x:
                knowledge_ban_edge = False
                #sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True

                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        # append_value(cg.sepset, x, y, ())
                        # append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))  # after all conditioning sets at
                        edge_removal.append((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    p = cg.ci_test(x, y, S)
                    # update Pmax
                    if p > Pmax['({},{})'.format(x, y)]:
                        Pmax['({},{})'.format(x, y)] = p

                    # if all elements of Pmax are not -1, run lines 11-20
                    if not (-1 in Pmax.values()):
                        reject_id, accept_id = Benjamini_Yekutieli(list(Pmax.items()), alpha)

                        # remove those accept H0 (no edge)
                        for pair in accept_id:
                            start, end = pair.split(',')
                            start = int(start.replace('(', ''))
                            end = int(end.replace(')', ''))
                            if not stable:
                                edge1 = cg.G.get_edge(cg.G.nodes[start], cg.G.nodes[end])
                                if edge1 is not None:
                                    cg.G.remove_edge(edge1)
                                edge2 = cg.G.get_edge(cg.G.nodes[start], cg.G.nodes[end])
                                if edge2 is not None:
                                    cg.G.remove_edge(edge2)
                                # append_value(cg.sepset, start, end, S)
                                # append_value(cg.sepset, end, start, S)
                            else:
                                edge_removal.append((start, end))  # after all conditioning sets at
                                edge_removal.append((end, start))
                                # for s in S:
                                #    sepsets.add(s)

                        # if (a,b) has been removed, break the for loop at line 7
                        if '({},{})'.format(x, y) in accept_id:
                            break

                # append_value(cg.sepset, x, y, tuple(sepsets))
                # append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress:
        pbar.close()

    return cg
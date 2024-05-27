import numpy as np
import pandas as pd
import networkx as nx

from data.structual_equations import distribution,function
from timediscover.graph.NxGraph import get_parents,ftdag_to_mag


class TimeData:
    def __init__(self, ftdag, steq, k, n, **kwargs):
        # d - number of vertices
        # k - subsampling factor
        # n - sample size
        self.ftdag = ftdag
        self.ssteq, self.ftsteq = steq
        self.d = len(set([name.split('_')[0] for name in self.ftdag.nodes]))
        self.k = k
        self.n = n
        self.mag = ftdag_to_mag(self.ftdag, self.d, self.k)
        self.mag_ske = self.mag[0] + self.mag[1]  # skeleton_of_mag = direct edges + bidirect edges

        self.data = dict()
        self.generate(**kwargs)
        self.data = pd.DataFrame(self.data)
        self.data = self.data.reindex(columns=list(self.ftdag.nodes))

        self.observation = dict()
        for node in self.data.keys():
            var, time = node.split('_')
            time = int(time)
            if time == 0 or time == self.k + 1:
                self.observation[node] = self.data[node]
        self.observation = pd.DataFrame(self.observation)

    def generate(self, **kwargs):
        # V_i = sum_j f_j(PA_j) + exo_i
        for node in nx.topological_sort(self.ftdag):
            var, time = node.split('_')
            time = int(time)
            exodist = self.ssteq[var]
            parents = get_parents(self.ftdag, node)

            if len(parents) == 0:
                assert time == 0, 'Node {} at time {} has no parents.'.format(var, time)
                self.data[node] = distribution(exodist, **kwargs)(size=self.n)
            else:
                exo = distribution(exodist, **kwargs)(size=self.n)
                for parent in parents:
                    key = '{}->{}'.format(parent, node)
                    func = self.ftsteq[key]
                    exo += function(func, **kwargs)(self.data[parent])
                self.data[node] = exo
import numpy as np
import sys
sys.path.append('../../tools/')
from time_discover import time_discover
import networkx as nx

if __name__ == '__main__':
    dataset = np.load('./data/ADNI30T3729_BtoM6_d90.npy')
    d = dataset.shape[1] // 2; k = 1
    node_names = ['X{}_0'.format(ind + 1) for ind in range(d)] + ['X{}_{}'.format(ind + 1, k + 1) for ind in range(d)]

    sdag = time_discover(dataset, node_names, d, k, citest='fisherz',
                         real_world=True, cache_path='./cache', significance=0.05,depth=4)
    nx.write_gml(sdag, './result/ADNI30T3729_BtoM6_d{}_dag.gml'.format(d))
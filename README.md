## Proximal Causal Discovery

This repository contains the official implementation for the following papers:

- Mingzhou Liu et al. [Causal Discovery from Subsampled Time Series with Proxy Variables](https://arxiv.org/abs/2305.05276). NeurIPS 2023.
- Mingzhou Liu et al. [Causal Discovery via Conditional Independence Testing with Proxy Variables](https://arxiv.org/abs/2305.05281). ICML 2024.

### Requirements

numpy, pandas, networkx, r-base, ect. 

Install the dependencies by ```conda install --file requirements.txt```

### How to run (demo)

```
from timediscover.graph.NxGraph import erdoes_renyi
from data.data_gen_easy import TimeData
from timediscover.search.TD import td
from evaluation.evaluation import *

adj = erdoes_renyi(5,0.3) # generate random sDAG with 5 nodes and p=0.3
data_gen = TimeData(adj,k=1,n=200) # generate time-series data with sample_size=200 and subsampling_factor=2 
# (k=1 in the code means subsampling_factor=2 for the paper's notation)

# Recover the MAG and sDAG
mag,sdag = td(data = data_gen.observation.values,
              node_names = data_gen.observation.columns,
              indep_test = 'fisherz', proxy_test = 'proxy_test',
              alpha = 0.05, subsampling_factor=data_gen.k)
              
# One can specify indep_test = 'fisherz', 'kci', 'info_test', or 'd_separation' (for Oracle CI Test)
# and proxy_test = 'proxy_test' or 'd_separation' (for Oracle Proxy Test)
# To use the oracle test, one should provide true_dag=data_gen.ftdag in td()
# One can also specify more hyperparameters, e.g., those for proxy-test, see the comment of the td() function for details

# Evaluation
prec,rec = precision_skeleton(mag,data_gen.mag_ske),recall_skeleton(mag,data_gen.mag_ske)
print('Precision: {}, Recall: {} of MAG'.format(prec,rec))
prec,rec = precision(sdag.edges,data_gen.sdag.edges), recall(sdag.edges,data_gen.sdag.edges)
print('Precision: {}, Recall: {} of sDAG'.format(prec,rec))
```

If you want to use custom data, please re-format it to a pandas.DataFrame with colum names 

```['X1_0','X2_0',...,'Xd_0','X1_k','X2_k',...,'Xd_k']```

where $d$ is number of vertices in the summary graph and $k$ is the subsampling factor.

### File structure

The code structure takes reference from the [causal-learn](https://github.com/py-why/causal-learn/tree/main) package

- Code to causal discovery algorithm 

```timediscover.search```

- Code to proximal conditional independence testing

```timediscovery.utils.ProxyTest```

### Contact

liumingzhou@stu.pku.edu.cn

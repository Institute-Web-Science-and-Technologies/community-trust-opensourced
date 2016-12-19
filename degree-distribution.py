from __future__ import division

import sys
import igraph
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

filename = sys.argv[1]
df = pd.read_csv(filename, header=None)

nodes = np.unique(df[[0, 1]].values);
max_node_num = max(nodes) + 1
num_nodes = len(nodes)

G = igraph.Graph(directed=True)
G.add_vertices(max_node_num)
G.add_edges(df[[0, 1]].values)

G = G.subgraph(nodes)

degrees = G.degree()
bins = np.unique(degrees)
hist, _ = np.histogram(degrees, np.append(bins, max(bins)+1))

# plot
plt.loglog(bins, hist, '.')
plt.xlabel('Degree(d)')
plt.ylabel('Frequency')

plt.savefig('degree-distribution.eps', format='eps', dpi=1000)


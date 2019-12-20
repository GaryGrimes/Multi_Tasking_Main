#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:31:07 2019

@author: gary
"""
import numpy as np
from kmodes.kmodes import KModes

# reproduce results on small soybean data set
x = PT_dummy.loc[:, '1':'37'].dropna()


# k-modes 

kmodes_cao = KModes(n_clusters=4, init='Cao', verbose=2)
kmodes_cao.fit(x)

kmodes_huang = KModes(n_clusters=4, init='Huang', verbose=1)
kmodes_huang.fit(x)

# Print cluster centroids of the trained model.
print('k-modes (Cao) centroids:')
print(kmodes_cao.cluster_centroids_)
# Print training statistics
print('Final training cost: {}'.format(kmodes_cao.cost_))
print('Training iterations: {}'.format(kmodes_cao.n_iter_))

# Print cluster centroids of the trained model.
print('k-modes (Huang) centroids:')
print(kmodes_huang.cluster_centroids_)
# Print training statistics
print('Final training cost: {}'.format(kmodes_huang.cost_))
print('Training iterations: {}'.format(kmodes_huang.n_iter_))

#%% Cao's method
print('"Cao"s method')
pnt = []
for cent in kmodes_cao.cluster_centroids_:
    places = np.nonzero(cent)[0].tolist()
    p_semantic = []
    while places:
        p_semantic.append(Place_names[places.pop(0)+1])
    pnt.append(p_semantic)
while pnt:
    print(pnt.pop(0))

# Huang's method
print('"Huang"s method')
res2 = []
for cent in kmodes_huang.cluster_centroids_:
    places = np.nonzero(cent)[0].tolist()
    p_semantic = []
    while places:
        p_semantic.append(Place_names[places.pop(0)+1])
    res2.append(p_semantic)
while res2:
    print(res2.pop(0))
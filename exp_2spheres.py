"""
Active Domain Expansion for the high-dimensional two-sphere example

Author(s): Wei Chen (wchen459@umd.edu)
"""

import math
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.patches as patches
from scipy.stats import norm
from scipy.optimize import differential_evolution
import numdifftools as nd
from libact.base.dataset import Dataset
from libact.query_strategies import RandomSampling

from al_models import GPC, GPR
from query_strategies import UncertSampling, EpsilonMarginSampling

    
def get_label(D, landmark, threshold):
    
    L = np.min(pairwise_distances(D, landmark), axis=1) < threshold
    L = L*2-1
    return L
    
def gen_samples(d, bounds, density=500):
    
    r = bounds[1] - bounds[0]
    N = int(density * r[0] * r[1] * r[2])
    samples = np.random.uniform(bounds[0], bounds[1], (N, d))
    
    return samples
    
def expand_pool(D, bounds_old, expansion_rate):
    
    d = D.shape[1]

    # Expand the previous query boundary based on D
    bounds_new = np.zeros_like(bounds_old)
    bounds_new[0,:] = np.min(D, axis=0) - expansion_rate
    bounds_new[1,:] = np.max(D, axis=0) + expansion_rate
    
    # Generate samples inside the new boundary
    pool = gen_samples(d, bounds_new)
    
    # Exclude samples inside the old boundary
    indices = np.logical_or(pool[:,0] < bounds_old[0,0], pool[:,0] > bounds_old[1,0])
    indices = np.logical_or(indices, pool[:,1] < bounds_old[0,1])
    indices = np.logical_or(indices, pool[:,1] > bounds_old[1,1])
    pool = pool[indices]
#    print pool.shape[0]
    
    return pool, bounds_new
        
    
if __name__ == "__main__":
    
    n_iter = 160
    d = 3
    
    length_scale = .5
    margin = .5 # higher -> emphasize more on variance -> more exploration/less exploit
    eta = .9
    
    # Set a global boundary
    BD = np.array([[-2, -2, -2], 
                   [5, 2, 2]])

    # Create experimental dataset
    landmark = np.array([[0, 0, 0], 
                         [3, 0, 0]])
        
    # Initial labeled samples
    np.random.seed(0)
    D0 = np.random.rand(10, d)
    D = D0
    threshold = 1.
    L = get_label(D, landmark, threshold)
    print L
    dataset = Dataset(D, L)
    
    # Generate test set
    D_test = gen_samples(d, BD, density=100)
    L_test = get_label(D_test, landmark, threshold)
    testset = Dataset(D_test, L_test)
    
    sigma = np.mean(pairwise_distances(D0))
    
    qs = EpsilonMarginSampling(
         dataset, # Data25set object
         model=GPC(RBF(length_scale), optimizer=None),
#         model=GPC(RBF(1, (.5,5)), n_restarts_optimizer=5),
         margin=margin,
         eta=eta
         )
    
    qs1 = UncertSampling(
          dataset, # Dataset object
          model=GPC(RBF(length_scale), optimizer=None),
          method='sm'
          )
    
    qs2 = RandomSampling(dataset)

    center0 = np.mean(D[L==1], axis=0)
    center = center0
    bounds_old = np.vstack((np.min(D0, axis=0), np.max(D0, axis=0)))
    acc = []
    f1s = []
    i = 0
    clf = GPC(RBF(length_scale), optimizer=None)
                    
    while i < n_iter+1:
        
        print 'Iteration: %d/%d' %(i, n_iter)
        
        # Generate a pool and expand dataset
        pool, bounds_new = expand_pool(D, bounds_old, expansion_rate=length_scale)
        for entry in pool:
            dataset.append(entry)
        
        # Query a new sample
        ask_id, clf = qs.make_query(center)
#        ask_id, clf = qs1.make_query()
#        ask_id = qs2.make_query()
#        clf.train(dataset)
        new = dataset.data[ask_id][0].reshape(1,-1)
#        print new
        
        # Update model and dataset
        l = get_label(new, landmark, threshold)
        dataset.update(ask_id, l) # update dataset
        D = np.vstack((D, new))
        L = np.append(L, l)
        
        # Compute the test accuracy
        acc.append(clf.score(testset))
        L_pred = clf.predict(D_test)
        f1s.append(f1_score(L_test, L_pred))
            
        if np.any(np.array(L[-10:]) == 1) and np.any(np.array(L[-10:]) == -1):
            center = D[np.array(L) == 1][-1] # the last positive sample
        else:
            center = center0
        
        i += 1
        bounds_old = bounds_new
#        expansion_rate = clf.get_kernel().get_params()['length_scale']
    
    plt.figure()
    ax = plt.subplot(111, aspect=1, projection = '3d')
    ax.set_xlim(bounds_new[:,0])
    ax.set_ylim(bounds_new[:,1])
    ax.set_zlim(bounds_new[:,2])
    ax.scatter(D[:, 0], D[:, 1], D[:, 2], s=20, c=L)
        
    plt.figure()
    plt.scatter(D_test[:,0], D_test[:,1], s=20, c=L_test, cmap=plt.cm.Paired)
    
    plt.figure()
    plt.plot(f1s)
    plt.ylim(0,1)
    plt.show()

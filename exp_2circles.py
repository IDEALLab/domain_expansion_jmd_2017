"""
Active Domain Expansion for the two-circle example

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
    
    L = (np.min(pairwise_distances(D, landmark), axis=1) < threshold)
    L = L*2-1
    return L
    
def gen_samples(d, bounds, density=50):
    
    r = bounds[1,:] - bounds[0,:]
    N = int(density * r[0] * r[1])
    samples = np.random.rand(N, d)
    samples = samples * r.reshape(1, d) + bounds[0,:].reshape(1, d)
    
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
    
    n_iter = 95
    interval = 6
    d = 2
    
    length_scale = 1.
    margin = .7 # higher -> emphasize more on variance -> more exploration/less exploit
    
    # Set a global boundary
    BD = np.array([[-8, -7], 
                   [8, 9]])
    
    # Create experimental dataset
    landmark = np.array([[-2, 4], 
                         [2, 0]])
        
    # Initial labeled samples
    np.random.seed(0)
    D0 = np.random.rand(5, d)
    D = D0
    threshold = 1.5
    L = get_label(D, landmark, threshold)
    print L
    dataset = Dataset(D, L)
    
    # Generate test set
    D_test = gen_samples(d, BD, density=10)
    L_test = get_label(D_test, landmark, threshold)
    testset = Dataset(D_test, L_test)
    
    sigma = np.mean(pairwise_distances(D0))
    
    qs = EpsilonMarginSampling(
         dataset, # Dataset object
         model=GPC(RBF(length_scale), optimizer=None),
#         model=GPC(RBF(1, (.5,5)), n_restarts_optimizer=5),
         margin=margin
         )
    
    qs1 = UncertSampling(
          dataset, # Dataset object
          model=GPC(RBF(length_scale), optimizer=None),
          method='sm'
          )
    
    qs2 = RandomSampling(dataset)
    
    # Arrange subplots
    im_rows = math.floor((1+n_iter/interval)**0.5)
    im_cols = math.ceil(float(1+n_iter/interval)/im_rows)
    plt.figure(figsize=(im_cols*10, im_rows*7))
    
    # Create a mesh grid
    xx, yy = np.meshgrid(np.linspace(BD[0][0], BD[1][0], 200),
                         np.linspace(BD[0][1], BD[1][1], 200))

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

        if i%interval == 0:
            # Plot the decision function for each datapoint on the grid
            grid = np.vstack((xx.ravel(), yy.ravel())).T
            Z0 = clf.predict_real(grid)[:,-1] # to show probability
            Z0 = Z0.reshape(xx.shape)
            Z1, Z4 = clf.predict_mean_var(grid) # to show posterior mean and variance
            Z1 = Z1.reshape(xx.shape)
            Z4 = Z4.reshape(xx.shape)
            b = clf.get_kernel().get_params()['length_scale']
            Z2 = norm.cdf(-(np.abs(Z1)+margin), 0, np.sqrt(Z4)) # to show query boundary
            Z2 = Z2.reshape(xx.shape)
            Z3 = get_label(grid, landmark, threshold) # to show ground truth decision boundary
            Z3 = Z3.reshape(xx.shape)
            
            ax = plt.subplot(im_rows, im_cols, i/interval+1)
#            image = plt.imshow(Z1, interpolation='nearest',
#                               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#                               aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
#            plt.colorbar(image)
            plt.contour(xx, yy, Z1, levels=[0], linewidths=2, linetypes='--', c='g', alpha=0.5) # estimated decision boundary
            plt.contour(xx, yy, Z2, levels=[0.8*norm.cdf(-margin)], linewidths=1, linetypes=':', c='b', alpha=0.3)
            image = plt.imshow(Z3<0, interpolation='nearest',
                               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                               aspect='auto', origin='lower', cmap=plt.get_cmap('gray'), alpha=.5) # ground truth domain
            plt.scatter(new[0,0], new[0,1], s=130, c='y', marker='*')
            plt.scatter(D[:, 0], D[:, 1], s=30, c=L, cmap=plt.cm.Paired)
            plt.scatter(center[0], center[1], s=100, c='y', marker='+')
#            plt.scatter(D0[:, 0], D0[:, 1], s=10, c='k')
#            for entry in dataset.get_unlabeled_entries():
#                plt.scatter(entry[1][0], entry[1][1], color='k', alpha=0.2)
            ax.add_patch(
                patches.Rectangle(
                    tuple(bounds_new[0]),
                    bounds_new[1,0]-bounds_new[0,0],
                    bounds_new[1,1]-bounds_new[0,1],
                    fill=False,
                    linestyle='dashed'
                )
            )
#            plt.title("Iteration: %d" % i, fontsize=20)
            plt.title("%d : %s\n Log-Marginal-Likelihood:%.3f"
                  % (i, clf.get_kernel(), clf.get_log_marginal_likelihood()), fontsize=12)
            plt.xticks(())
            plt.yticks(())
            plt.tight_layout()
        
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
    plt.scatter(D_test[:,0], D_test[:,1], s=20, c=L_test, cmap=plt.cm.Paired)
    plt.figure()
    plt.plot(f1s)
    plt.title('F1 score')
    plt.show()

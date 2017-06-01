"""
Active Domain Expansion for the novelty discovery examples

Author(s): Wei Chen (wchen459@umd.edu)
"""

import math
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams.update({'font.size': 20})

from filters import no_intersect
from diverse_selection import select_diverse
from synthesis import synthesize_shape, save_plot
from util import load_array, save_model
from query_strategies import UncertSampling, EpsilonMarginSampling
from al_models import GPC, GPR
from shape_plot import plot_shape

from libact.base.dataset import Dataset
from libact.labelers import InteractiveLabeler
from libact.query_strategies import RandomSampling

    
def gen_intersect(n, d, mn, mx, model_name='PCA', c=0):
    
    i = 0
    samples = []
    r = mx - mn
    while i < n:
        sample = np.random.rand(d) * r + mn
        sample = sample.reshape(-1,d)
        X = synthesize_shape(sample, c=c, model_name=model_name, raw=True)
        if not no_intersect(X):
            i += 1
            samples.append(sample[0])
            
    return np.array(samples)
    
    
class Interface(InteractiveLabeler):
    
    def label(self, feature):

        banner = "Enter the associated label with the image: "
        
        if self.label_name is not None:
            banner += str(self.label_name) + ' '
                    
        while True:
            try:
                lbl = input(banner)
                break
            except ValueError:
                print("Oops!  Invalid number.  Try again...")
            except NameError:
                print("Oops!  Invalid number.  Try again...")
            except SyntaxError:
                print("Oops!  Invalid number.  Try again...")

        return self.label_name.index(lbl)
        
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
    print pool.shape[0]
    
    return pool, bounds_new
    
    
if __name__ == "__main__":

    np.random.seed(0)

    model_name = 'PCA'
    c = 0
    
    n_iter = 1000
    
    length_scale = 1.3
    margin = .7 # higher -> emphasize more on variance -> more exploration/less exploit
    
    T = load_array(model_name+'_f', c) # Training samples
    d = T.shape[1]
    
    # Set a global boundary
    BD = np.array([[-10, -30, -30], 
                   [60, 30, 30]])
    
    # Initial valid samples
    kv = 10 # number of initial valid samples
    Dv = T[select_diverse(T, kv)]
    Xv = synthesize_shape(Dv, c=c, model_name=model_name, raw=True)
    save_plot(Dv, Xv, c=c, model_name=model_name+'_initial')
    # Initial invalid samples
    ki = 1 # number of initial invalid samples
    mn = np.min(T, axis=0)
    mx = np.max(T, axis=0)
    Di = gen_intersect(ki, d, mn, mx)
    
    # Concatenate valid and invalid
    D0 = np.vstack((Dv, Di))
    D = D0
    L = -np.ones(kv+ki, dtype=int)
    L[:kv] = np.ones(kv, dtype=int)
    dataset = Dataset(D, L)
    
    qs = EpsilonMarginSampling(
         dataset, # Dataset object
         model=GPC(RBF(length_scale), optimizer=None),
         margin=margin
         )
    
    qs1 = UncertSampling(   
          dataset, # Dataset object
          model=GPC(RBF(length_scale), optimizer=None),
          method='sm'
          )
    
    qs2 = RandomSampling(dataset)
    
    lbr = Interface(label_name=[0, 1])
    
    # Create a mesh grid
    xx, yy = np.meshgrid(np.linspace(BD[0][0], BD[1][0], 200),
                         np.linspace(BD[0][1], BD[1][1], 200))
    
    center0 = np.mean(D[L==1], axis=0)
    center = center0
    bounds_old = np.vstack((np.min(D0, axis=0), np.max(D0, axis=0)))
    i = 0
                    
    while i < n_iter+1:
        
        print 'Iteration: %d/%d' %(i, n_iter)
        
        # Generate a pool and expand dataset
        pool, bounds_new = expand_pool(D, bounds_old, expansion_rate=length_scale)
        for entry in pool:
            dataset.append(entry)
        
        # Query a new sample
        ask_id, clf = qs.make_query(center)
        new = dataset.data[ask_id][0].reshape(1,-1)

        X = synthesize_shape(new, c=0, model_name=model_name, raw=True)
        
        if no_intersect(X):
            
            fig = plt.figure(figsize=(16,8), tight_layout=True)
            fig.patch.set_facecolor('w')
            plt.suptitle("Iteration: %d" % i, fontsize=30)
            
            if d == 2:
                # Plot the decision function for each datapoint on the grid
                grid = np.vstack((xx.ravel(), yy.ravel())).T
                Z1, Z4 = clf.predict_mean_var(grid) # to show posterior mean and variance
                Z1 = Z1.reshape(xx.shape)
                Z4 = Z4.reshape(xx.shape)
                b = clf.get_kernel().get_params()['length_scale']
#                print clf.get_kernel()
                Z2 = norm.cdf(-(np.abs(Z1)+margin), 0, np.sqrt(Z4)) # to show query boundary
                Z2 = Z2.reshape(xx.shape)
                
                ax = plt.subplot(121, aspect=1)
#                image = plt.imshow(Z1, interpolation='nearest',
#                                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#                                   aspect=1, origin='lower', cmap=plt.cm.PuOr_r)
#                plt.colorbar(image)
                plt.contour(xx, yy, Z1, levels=[0], linewidths=2, linetypes='--', c='b', alpha=0.5) # decision boundary
#                plt.contour(xx, yy, Z2, levels=[0], linewidths=1, linetypes=':', c='r', alpha=0.3)
                plt.scatter(new[0, 0], new[0, 1], s=150, c='y', marker='*')
                plt.scatter(D[:, 0], D[:, 1], s=10, c=L, cmap=plt.cm.Paired)
#                plt.scatter(D0[:, 0], D0[:, 1], s=10, c='k')
#                for entry in dataset.get_unlabeled_entries():
#                    plt.scatter(entry[1][0], entry[1][1], s=10, color='k', alpha=0.3)
                ax.add_patch(
                    patches.Rectangle(
                        tuple(bounds_new[0]),
                        bounds_new[1,0]-bounds_new[0,0],
                        bounds_new[1,1]-bounds_new[0,1],
                        fill=False,
                        linestyle='dashed'
                    )
                )
                plt.xticks(())
                plt.yticks(())
            
            else:
                ax = plt.subplot(121, aspect=1, projection = '3d')
                ax.scatter(new[0, 0], new[0, 1], new[0, 2], s=150, c='y', marker='*')
                ax.scatter(D[:, 0], D[:, 1], D[:, 2], s=10, c=L, cmap=plt.cm.Paired)
                ax.set_xlim(BD[:,0])
                ax.set_ylim(BD[:,1])
                ax.set_zlim(BD[:,2])
            
            ax_glass = plt.subplot(122, aspect=1)
            plot_shape(X, 0, 0, ax_glass, mirror=True)
            plt.setp(ax_glass.spines.values(), color='w')
            plt.xticks(())
            plt.yticks(())
            plt.show(block=False)
                
            l = lbr.label(new)
            
        else:
            l = 0 # if self-intersect, automatically label 0
            
        l = l*2-1 # get label
        dataset.update(ask_id, l) # update dataset
    
        D = np.vstack((D, new))
        L = np.append(L, l)
        
        if np.any(np.array(L[-10:]) == 1):
            center = D[np.array(L) == 1][-1] # the last positive sample
        else:
            center = center0
        
        i += 1
        bounds_old = bounds_new
        
        plt.close()
        save_model(clf, model_name+'_clf', c)

    
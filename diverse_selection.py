import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from dpp import sample_dpp


def select_diverse(X, k):
    
    m = X.shape[0]
    sigma = 0.1*np.mean(pairwise_distances(X)+np.identity(m))
    
    sample = [np.random.choice(m)] # randomly select the first sample
    for i in range(1, k):
        X_selected = X[sample].reshape((i,-1))
        kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(X_selected)
        kde_scores = np.exp(kde.score_samples(X))
        sample.append(np.argmin(kde_scores))
        
    return sample

    
def test(k, sampling='grid'):
    
    from itertools import product
    
    if sampling == 'gaussian':
        X = np.random.normal(scale=.1, size=(1000, 2))
    
    else:
        # Genetate grid
        x = np.arange(0, 1.1, 0.1)
        y = np.arange(0, 1.1, 0.1)
        X = np.array(list(product(x, y)))
    
    sample = select_diverse(X, k)
    M = np.exp(-pairwise_distances(X)**2/(10./k)**2)
    dpp = sample_dpp(M, k)
    rand = np.random.choice(X.shape[0], k)
    
    # Plot results
    mn = np.min(X, axis=0)-0.1
    mx = np.max(X, axis=0)+0.1
    plt.figure()
    plt.subplot(131)
    plt.plot(X[sample,0],X[sample,1],'o',)
    plt.plot(X[:,0],X[:,1],'g.', alpha=0.5)
    plt.title('Sample from the KDE')
    plt.xlim(mn[0], mx[0])
    plt.ylim(mn[1], mx[1])
    plt.subplot(132)
    plt.plot(X[dpp,0],X[dpp,1],'o',)
    plt.plot(X[:,0],X[:,1],'g.', alpha=0.5)
    plt.title('Sample from the k-DPP')
    plt.xlim(mn[0], mx[0])
    plt.ylim(mn[1], mx[1])
    plt.subplot(133)
    plt.plot(X[rand,0],X[rand,1],'o',)
    plt.plot(X[:,0],X[:,1],'g.', alpha=0.5)
    plt.title('Random sampling')
    plt.xlim(mn[0], mx[0])
    plt.ylim(mn[1], mx[1])
    
    
if __name__ == "__main__":
    
    k = 10
    
    test(k, sampling='gaussian')

#    from synthesis import synthesize_shape, save_plot
#    
#    a = 0.1
#    A = (1+2*a)*np.random.rand(1000,3)-a # Specify shape attributes here
#    model_name = 'PCA'
#    c = 0
#    
#    X, indices = synthesize_shape(A, c=0, model_name='PCA')
#    A = A[indices] # set of valid attributes
#    X = X[indices] # set of valid shapes
#    
#    M = np.exp(-pairwise_distances(A)**2/sigma**2) # gaussian kernel
#    Y = sample_dpp(M, k)
#    
#    save_plot(A[Y], X[Y], c=c, model_name=model_name)
    
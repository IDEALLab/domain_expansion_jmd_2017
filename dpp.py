import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import ConfigParser
from shape_plot import plot_samples


def elem_sympoly(lambdas, k):
    ''' Given a vector of lambdas and a maximum size k, determine the value of
        the elementary symmetric polynomials:
        E(l+1,n+1) = sum_{J \subseteq 1..n,|J| = l} prod_{i \in J} lambda(i) '''
  
    N = len(lambdas)
    E = np.zeros((k+1, N+1))
    E[0,:] = np.ones(N+1)
    for l in range(1, k+1):
        for n in range(1, N+1):
            E[l,n] = E[l,n-1] + lambdas[n-1]*E[l-1,n-1]
    
    return E
  
  
def sample_k(lambdas, k):
    ''' Pick k lambdas according to p(S) \propto prod(lambda \in S) '''
    
    # compute elementary symmetric polynomials
    E = elem_sympoly(lambdas, k)

    # iterate
    i = len(lambdas)-1
    remaining = k
    S = np.zeros(k, dtype=int)
    while remaining > 0:
        
        # compute marginal of i given that we choose remaining values from 0:i
        if i == remaining-1:
            marg = 1
        else:
            marg = lambdas[i] * E[remaining-1,i] / E[remaining,i+1]
        
        # sample marginal
        if np.random.rand(1) < marg:
            S[remaining-1] = i
            remaining -= 1

        i -= 1
    
    return S
    

def decompose_kernel(M):
    
    D, V = np.linalg.eig(M)
    V = np.real(V)
    D = np.real(D)
    
    sort_perm = D.argsort()
    D.sort()
    V = V[:, sort_perm]
    
    return V, D


def sample_dpp(M, k=None):
    ''' Sample a set Y from a dpp.  M is a kernel, and k is (optionally)
        the size of the set to return. '''
    
    V, D = decompose_kernel(M)
    
    if k is None:
        # choose eigenvectors randomly
        D = D / (1+D)
        v = np.nonzero(np.random.rand(len(D)) <= D)[0]
    else:
        # k-DPP
        v = sample_k(D, k)
      
    k = len(v)
    V = V[:,v]
    
    # iterate
    Y = np.zeros(k, dtype=int)
    for i in range(k-1, -1, -1):
      
        # compute probabilities for each item
        P = np.sum(V**2, axis=1)
        P = P / np.sum(P)
    
        # choose a new item to include
        Y[i] = np.nonzero(np.random.rand(1) <= np.cumsum(P))[0][0]
    
        # choose a vector to eliminate
        j = np.nonzero(V[Y[i],:])[0][0]
        Vj = V[:, j]
        V = np.delete(V, j, axis=1)
    
        # update V
        V = V - Vj[:, None] * V[Y[i],:][None, :] / Vj[Y[i]]
    
        # orthogonalize
        for a in range(i):
            for b in range(a):
                V[:,a] = V[:,a] - np.inner(V[:,a], V[:,b]) * V[:,b]
            V[:,a] = V[:,a] / np.linalg.norm(V[:,a])
    
    Y = np.sort(Y)

    return Y

    
def test(k, sigma, sampling='grid'):
    
    import matplotlib.pyplot as plt
    from itertools import product
    
    if sampling == 'gaussian':
        X = np.random.normal(scale=.1, size=(1000, 2))
    
    else:
        # Genetate grid
        x = np.arange(0, 1.1, 0.1)
        y = np.arange(0, 1.1, 0.1)
        X = np.array(list(product(x, y)))
    
    M = np.exp(-pairwise_distances(X)**2/sigma**2) # gaussian kernel
    sample = sample_dpp(M, k)
    rand = np.random.choice(X.shape[0], k)
    
    # Plot results
    mn = np.min(X, axis=0)-0.1
    mx = np.max(X, axis=0)+0.1
    plt.figure()
    plt.subplot(131)
    plt.plot(X[:,0],X[:,1],'.',)
    plt.title(sampling)
    plt.xlim(mn[0], mx[0])
    plt.ylim(mn[1], mx[1])
    plt.subplot(132)
    plt.plot(X[sample,0],X[sample,1],'.',)
    plt.title('Sample from the DPP')
    plt.xlim(mn[0], mx[0])
    plt.ylim(mn[1], mx[1])
    plt.subplot(133)
    plt.plot(X[rand,0],X[rand,1],'.',)
    plt.title('Random sampling')
    plt.xlim(mn[0], mx[0])
    plt.ylim(mn[1], mx[1])
    
    
if __name__ == "__main__":
    
    sigma = 0.1
    k = 100
    
    test(k, sigma, sampling='gaussian')

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
    
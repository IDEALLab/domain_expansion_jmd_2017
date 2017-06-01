"""
Synthesizes new shapes using trained models

Author(s): Wei Chen (wchen459@umd.edu)
"""

import ConfigParser
import numpy as np
from util import load_model, get_fname, gen_samples
from shape_plot import plot_synthesis
from data_processing import inverse_features


def gen_valid(N, d, bounds, clf):
    ''' Generate valid samples according to the classifier clf '''
    
#    A = []
#    i = 0
#    while i < N:
#        a = gen_samples(1, d, bounds)
#        if clf.predict(a) == np.array([1]):
#            A.append(a.flatten())
#            i += 1
#
#    return np.array(A)

    xx, yy = np.meshgrid(np.linspace(bounds[0][0], bounds[1][0], 21),
                         np.linspace(bounds[0][1], bounds[1][1], 21))
    A = np.vstack((xx.ravel(), yy.ravel())).T
    L = clf.predict(A)
    return A[L==1]
            

def synthesize_shape(attributes, c=0, model_name='KPCA', raw=False):
    '''
    attributes : array_like
        Values of shape attributes which have the range [0, 1].
    model_name : str
        Name of the trained model.
    c : int
        The index of a cluster
    X : array_like
        Reconstructed high-dimensional design parameters
    '''
    
    if not raw:
        transforms = [load_model(model_name+'_fpca', c)]
        transforms.append(load_model(model_name+'_fscaler', c))
        raw_attr = inverse_features(attributes, transforms) # Min-Max normalization
        
    else:
        raw_attr = attributes
    
    model = load_model(model_name, c)
    xpca = load_model('xpca', c)
    dim_increase = xpca.inverse_transform
        
    data_rec = dim_increase(model.inverse_transform(raw_attr)) # Reconstruct design parameters
    
    return data_rec
    
def save_plot(attributes, data_rec, labels=None, c=0, model_name='KPCA'):
    
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    source = config.get('Global', 'source')
    
    # Get save directory
    save_dir = get_fname(model_name, c, directory='./synthesized_shapes/', extension='svg')
    
    print('Plotting synthesized shapes for %s_%d ... ' % (model_name, c))
    if source[:3] == 'rw-':
        X = plot_synthesis(attributes, data_rec, labels, save_dir, model_name)
    else:
        X = plot_synthesis(attributes, data_rec, labels, save_dir, model_name, mirror=False)
                              
#    np.save(get_fname(model_name, c, directory='./synthesized_shapes/', extension='npy'), X)
    
                                 
if __name__ == "__main__":
    
    model_name = 'PCA'
    c = 0
#    attributes = np.random.rand(10, 2) # Specify shape attributes here
#    BD = np.array([[-40, -30], 
#                   [30, 30]])
    BD = np.array([[-3, -2], 
                   [3, 3]])
    clf = load_model(model_name+'_clf', c)
    attributes = gen_valid(100, 2, BD, clf) # Specify shape attributes here
    data_rec = synthesize_shape(attributes, c=c, model_name=model_name, raw=False)
    save_plot(attributes, data_rec, c=c, model_name=model_name)
    
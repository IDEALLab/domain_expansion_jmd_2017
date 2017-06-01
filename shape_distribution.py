"""
Plots given shapes in the semantic space using trained models

Author(s): Wei Chen (wchen459@umd.edu)
"""

import ConfigParser
import numpy as np
from util import load_model, get_fname
from shape_plot import plot_samples

model_name = 'KPCA'
c = 0

config = ConfigParser.ConfigParser()
config.read('config.ini')
source = config.get('Global', 'source')

directory = './shape_distribution/' + source + '/'

indices = np.load(directory+'ranking.npy')
indices = indices.flatten()[:5]
X = np.load(get_fname(model_name, c, directory=directory, extension='npy'))[indices] # Shapes to plot

# Get the low-dimensional representation of X
xpca = load_model('xpca', c)
F = xpca.transform(X)

model = load_model(model_name, c)
F = model.transform(F)

fpca = load_model(model_name+'_fpca', c)
F = fpca.transform(F)
fscaler = load_model(model_name+'_fscaler', c)
F = fscaler.transform(F)

if source[:3] == 'rw-':
    plot_samples(F, X, None, range(X.shape[0]), [], directory, '', c, annotate=True)
else:
    plot_samples(F, X, None, range(X.shape[0]), [], directory, '', c, mirror=False, annotate=True)
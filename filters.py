"""
Filters for selecting valid shapes

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np

def remove_outside(attributes, boundary):
    ''' Remove the samples outside the boundary '''
    
    indices = []
    for i in range(len(attributes)):
        c = tuple(attributes[i]) + (1,)
        c = np.expand_dims(c, axis=1)
        e = np.dot(boundary, c)
        if np.all(e <= 0):
            indices.append(i)
    return indices

def no_intersect(shapes):
    ''' The x coordinates of shape contours cannot be positive '''
    
    m = shapes.shape[0]
    filt1 = np.all(shapes[:,::2] < 0, axis=1)
    filt2 = np.all(shapes[:,::2] > 0, axis=1)
    filt = filt1 + filt2
    if m == 1:
        return filt[0]
    else:
        return np.arange(m)[filt].tolist()
    
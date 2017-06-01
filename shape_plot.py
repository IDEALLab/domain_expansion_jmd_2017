"""
Plots samples or new shapes in the semantic space.

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
"""

from matplotlib import pyplot as plt
from sklearn import preprocessing
import numpy as np
import itertools
from data_processing import inverse_features
from filters import no_intersect, remove_outside

def plot_shape(xys, attribute_x, attribute_y, ax, mirror, linewidth=1.5, color='blue', alpha=1, scale=.07):
    
    m = xys.reshape(-1,2)
#    mx = max([y for (x, y) in m])
#    mn = min([y for (x, y) in m])
    xscl = scale# / (mx - mn)
    yscl = scale# / (mx - mn)
    ax.plot( *zip(*[(x * xscl + attribute_x, -y * yscl + attribute_y)
                       for (x, y) in m]), linewidth=linewidth, color=color, alpha=alpha)
    if mirror:
        ax.plot( *zip(*[(-x * xscl + attribute_x, -y * yscl + attribute_y) 
                       for (x, y) in m]), linewidth=linewidth, color=color, alpha=alpha)
        plt.fill_betweenx( *zip(*[(-y * yscl + attribute_y, -x * xscl + attribute_x, x * xscl + attribute_x)
                           for (x, y) in m]), color=color, alpha=alpha*.8)

def plot_samples(features, data, data_rec, train, test, save_path, model_name, cluster, mirror=True, annotate=False):
    
    ''' Create 3D scatter plot and corresponding 2D projections
        of at most the first 3 dimensions of data'''
    
    plt.rc("font", size=font_size)
    n_samples_train = len(train)
    n_samples_test = len(test)
    n_dim = features.shape[1]
    
    if n_dim == 1:
        features = np.concatenate((features, np.zeros_like((features))), axis=1)
        n_dim = 2
    
    if n_dim == 3:
        # Create a 3D scatter plot
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection = '3d')
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([features[:,0].max()-features[:,0].min(), features[:,1].max()-features[:,1].min(), 
                              features[:,2].max()-features[:,2].min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(features[:,0].max()+features[:,0].min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(features[:,1].max()+features[:,1].min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(features[:,2].max()+features[:,2].min())
        ax3d.scatter(Xb, Yb, Zb, c='white', alpha=0)
        ax3d.scatter(features[:,0], features[:,1], features[:,2])
        ax3d.set_title(model_name)
        plt.savefig(save_path+model_name+'/'+str(cluster)+'_'+'3d.png', dpi=600)
        plt.close()
    
    # Project 3D plot to 2D plots and label each point
    figs = []
    ax = []
    k = 0
    for i in range(0, n_dim-1):
        for j in range(i+1, n_dim):
            figs.append(plt.figure())
            ax.append(figs[k].add_subplot(111))
            
            # Plot training data
            for index in range(n_samples_train):
                if annotate:
                    label = '{0}'.format(index+1)
                    plt.annotate(label, xy = (features[train][index,i], features[train][index,j]), size=10)
                ax[k].scatter(features[train][index,i], features[train][index,j], s = 7)
                plot_shape(data[train][index], features[train][index,i], features[train][index,j], ax[k],
                           mirror, color='red', alpha=.7)
                
                if data_rec is not None:
                    # Draw reconstructed samples for training data
                    plot_shape(data_rec[train][index], features[train][index,i], features[train][index,j], ax[k], 
                               mirror, color='green', alpha=.5)
            
            if len(test) != 0:
                #Plot testing data
                for index in range(n_samples_test):
                    if annotate:
                        label = '{0}'.format(index+1)
                        plt.annotate(label, xy = (features[test][index,i], features[test][index,j]), size=10)
                    ax[k].scatter(features[test][index, i], features[test][index, j], s = 7)
                    plot_shape(data[test][index], features[test][index,i], features[test][index,j], ax[k], 
                               mirror, color='blue', alpha=.7)
                    
                    if data_rec is not None:
                        # Draw reconstructed samples for testing data
                        plot_shape(data_rec[test][index], features[test][index,i], features[test][index,j], ax[k], 
                                   mirror, color='cyan', alpha=.7)            
                
            ax[k].set_title(model_name)
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)
            plt.xlabel('Dimension-'+str(i+1))
            plt.ylabel('Dimension-'+str(j+1))
            
            #ax[k].text(-0.1, -0.1, 'training error = '+str(err_train)+' / testing error = '+str(err_test))
            
            k += 1
            plt.tight_layout()
            plt.savefig(save_path+model_name+'/'+str(cluster)+'_'+str(i+1)+'-'+str(j+1)+'.png', dpi=600)
            plt.close()

def plot_grid(points_per_axis, n_dim, inverse_transform, dim_increase, transforms, save_path, model_name, 
              cluster, boundary=None, kde=None, mirror=True):
    
    ''' Uniformly plots synthesized shape contours in the semantic space.
        If the semantic space is 3D (i.e., n_dim=3), plot one slice of the 3D space at each time. '''
    
    plt.rc("font", size=font_size)
    lincoords = []
    lb = -1
    rb = 2
    
    for i in range(0,n_dim):
        lincoords.append(np.linspace(lb, rb, points_per_axis))
    coords_norm = list(itertools.product(*lincoords)) # Create a list of coordinates in the semantic space
    m = len(coords_norm)
    
    coords = inverse_features(coords_norm, transforms) # Min-Max normalization
    data_rec = dim_increase(inverse_transform(np.array(coords))) # Reconstruct design parameters
    if kde is not None:
        # Density evaluation for coords_norm
        kde_scores = np.exp(kde.score_samples(coords_norm))
    else:
        kde_scores = np.ones(m)
    
    # Filter invalid shapes
    indices = range(m)
    if mirror:
        indices = no_intersect(data_rec)
    if boundary is not None:
        indices1 = remove_outside(coords_norm, boundary)
        indices = list(set(indices) & set(indices1))
    
    if n_dim == 1:
        # Create a 1D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
#            ax.scatter(coords_norm[i][0], 0, s = 7)
            alpha = min(1, kde_scores[i] + .3)
            plot_shape(data_rec[i], coords_norm[i][0], 0, ax, mirror, linewidth=linewidth, 
                       alpha=alpha, scale=1.5/points_per_axis)
        
        ax.set_title(model_name)
        plt.xlim(lb-.1, rb+.1)
        plt.ylim(lb-.1, rb+.1)
        plt.axis('equal')
        plt.xlabel('Dimension-1')
        plt.ylabel('Dimension-2')
        plt.tight_layout()
        plt.savefig(save_path+model_name+'/'+str(cluster)+'_grid.png', dpi=600)
        
        plt.close()
        
    elif n_dim == 2:
        # Create a 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
#            ax.scatter(coords_norm[i][0], coords_norm[i][1], s = 7)
            alpha = min(1, kde_scores[i] + .3)
            plot_shape(data_rec[i], coords_norm[i][0], coords_norm[i][1], ax, 
                       mirror, linewidth=linewidth, alpha=alpha, scale=1.5/points_per_axis)
        
        ax.set_title(model_name)
        plt.xlim(lb-.1, rb+.1)
        plt.ylim(lb-.1, rb+.1)
        plt.xlabel('Dimension-1')
        plt.ylabel('Dimension-2')
        plt.tight_layout()
        plt.savefig(save_path+model_name+'/'+str(cluster)+'_grid.png', dpi=600)
        
#        if kde is not None:
#            for i in indices:
#                # Compute and annotate sparsity for coords_norm[i]
#                #kde_score = np.exp(kde.score_samples(np.reshape(coords_norm[i], (1, -1))))[0]
#                ax.annotate('{:.2f}'.format(kde_scores[i]), (coords_norm[i][0], coords_norm[i][1]), fontsize=12)
#            plt.tight_layout()
#            plt.savefig(save_path+model_name+'/'+str(cluster)+'_grid_sparsity.png', dpi=600)
        plt.close()
        
    elif n_dim == 3:
        # Create slices of 2D plots for n_dim = 3
        k = 0
        figs = []
        ax = []
        figs.append(plt.figure())
        ax.append(figs[k].add_subplot(111))
        xx = coords_norm[indices[0]][0]
        for i in indices:
                        
            if coords_norm[i][0] != xx:
                ax[k].set_title('%s (x = %.4f)' %(model_name, xx))
                plt.xlim(lb-.1, rb+.1)
                plt.ylim(lb-.1, rb+.1)
                plt.xlabel('Dimension-2')
                plt.ylabel('Dimension-3')
                plt.tight_layout()
                plt.savefig('%s%s/%d_grid_x=%.4f.png' % (save_path, model_name, cluster, xx), dpi=600)
                plt.close()
                
                k += 1
                xx = coords_norm[i][0]
                figs.append(plt.figure())
                ax.append(figs[k].add_subplot(111))
                
#            ax[k].scatter(coords_norm[i][1], coords_norm[i][2], s = 7)
            alpha = min(1, kde_scores[i] + .3)
            plot_shape(data_rec[i], coords_norm[i][1], coords_norm[i][2], ax[k], 
                       mirror, linewidth=linewidth, alpha=alpha, scale=1.5/points_per_axis)
        
        ax[k].set_title('%s (x = %.4f)' %(model_name, xx))
        plt.xlim(lb-.1, rb+.1)
        plt.ylim(lb-.1, rb+.1)
        plt.xlabel('Dimension-2')
        plt.ylabel('Dimension-3')
        plt.tight_layout()
        plt.savefig('%s%s/%d_grid_x=%.4f.png' % (save_path, model_name, cluster, xx), dpi=600)
        plt.close()
    
    else:
        print 'Cannot plot grid for semantic space dimensionality smaller than 1 or larger than 3!'
        
def plot_synthesis(attributes, data_rec, labels=None, save_path='', model_name='PCA', mirror=True):
    
    ''' Given shape attributes, plot synthesized shape contours in given locations of the semantic space. '''
    
    m = attributes.shape[0]
    n_dim = attributes.shape[1]

    plt.rc("font", size=font_size)    
    alpha = .5
    
    r = np.max(attributes, axis=0) - np.min(attributes, axis=0)
    mn = np.min(attributes, axis=0) - 0.1*r
    mx = np.max(attributes, axis=0) + 0.1*r

    colors = {-1:'black', 0:'red', 1:'blue'}
    if labels is None:
        labels = np.ones(m)
    
    if n_dim == 1:
        # Create a 1D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(m):
#            ax.scatter(attributes[i,0], 0, s = 7)
            plot_shape(data_rec[i], attributes[i,0], 0, ax, mirror, linewidth=linewidth, 
                       color=colors[labels[i]], scale=1./m)
        
        ax.set_title(model_name)
        plt.xlim(mn[0], mx[0])
        plt.ylim(-0.1, 0.1)
        plt.axis('equal')
        plt.xlabel('Dimension-1')
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        
        plt.close()
        
    elif n_dim == 2:
        # Create a 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.gca().set_aspect('equal', adjustable='box')
        
        for i in range(m):
#            plt.annotate(str(i), xy = (attributes[i,0], attributes[i,1]), size=10)
#            ax.scatter(attributes[i,0], attributes[i,1], s = 7)
            plot_shape(data_rec[i], attributes[i,0], attributes[i,1], ax, mirror, linewidth=linewidth, 
                       alpha=alpha, color=colors[labels[i]], scale=2*(mx[0]-mn[0])/m)
        
        ax.set_title(model_name)
        plt.xlim(mn[0], mx[0])
        plt.ylim(mn[1], mx[1])
        plt.xlabel('Dimension-1')
        plt.ylabel('Dimension-2')
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        
        plt.close()
        
    else:
        import math
        # Create a 2D plot
        points_per_axis = math.ceil(m**0.5)
        lb = -.5
        rb = 1.5
        lincoords = []
        for i in range(0,2):
            lincoords.append(np.linspace(lb, rb, points_per_axis))
        coords_norm = list(itertools.product(*lincoords))
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(m):
#            plt.annotate(str(i), xy = (coords_norm[i][0], coords_norm[i][1]), size=10)
#            ax.scatter(coords_norm[i][0], coords_norm[i][1], s = 7)
            plot_shape(data_rec[i], coords_norm[i][0], coords_norm[i][1], ax, mirror, 
                       linewidth=linewidth, scale=1./points_per_axis, color=colors[labels[i]])
        
        ax.set_title(model_name)
        plt.xlim(mn[0], mx[0])
        plt.ylim(mn[1], mn[1])
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        
        plt.close()
        
    if n_dim == 3:
        # Create a 3D scatter plot
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection = '3d')
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([attributes[:,0].max()-attributes[:,0].min(), attributes[:,1].max()-attributes[:,1].min(), 
                              attributes[:,2].max()-attributes[:,2].min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(attributes[:,0].max()+attributes[:,0].min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(attributes[:,1].max()+attributes[:,1].min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(attributes[:,2].max()+attributes[:,2].min())
        ax3d.scatter(Xb, Yb, Zb, c='white', alpha=0)
        for i in range(m):
            ax3d.scatter(attributes[i,0], attributes[i,1], attributes[i,2], color=colors[labels[i]])
        ax3d.set_title(model_name)
#        plt.savefig(save_path+'_3d.png', dpi=600)
        plt.show()
    

def plot_original_samples(points_per_axis, n_dim, inverse_transform, save_path, name,
                          variables, mirror=True):
    
    print "Plotting original samples ..."

    plt.rc("font", size=font_size)
    
    coords = variables
    coords_norm = preprocessing.MinMaxScaler().fit_transform(coords) # Min-Max normalization
    data_rec = inverse_transform(np.array(coords))
    indices = range(len(coords))

    if n_dim == 2:
        # Create a 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
            ax.scatter(coords_norm[i, 0], coords_norm[i, 1], s = 7)
            plot_shape(data_rec[i], coords_norm[i,0], coords_norm[i,1], ax, mirror, color='red', alpha=.7)

        ax.set_title(name)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('s')
        plt.ylabel('t')
        plt.tight_layout()
        plt.savefig(save_path+'original_samples.png', dpi=600)
        
        plt.close()
		
    else:
        print 'Cannot plot original samples for dimensionality other than 2!'
        
def plot_original_grid(points_per_axis, n_dim, min_maxes, inverse_transform, save_path, name, mirror=True):
    
    print "Plotting original grid ..."

    plt.rc("font", size=font_size)
    lincoords = []
    
    for i in range(0,n_dim):
        lincoords.append(np.linspace(min_maxes[i][0],min_maxes[i][1],points_per_axis))
    coords = list(itertools.product(*lincoords)) # Create a list of coordinates in the semantic space
    coords_norm = preprocessing.MinMaxScaler().fit_transform(coords) # Min-Max normalization
    data_rec = inverse_transform(coords)

    indices = range(len(coords))

    if n_dim == 2:
        # Create a 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
            ax.scatter(coords_norm[i, 0], coords_norm[i, 1], s = 7)
            plot_shape(data_rec[i], coords_norm[i,0], coords_norm[i,1], ax, mirror, linewidth=linewidth)

        ax.set_title(name)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('s')
        plt.ylabel('t')
        plt.tight_layout()
        plt.savefig(save_path+'original_grid.png', dpi=600)
        
        plt.close()
        
    else:
        print 'Cannot plot original grid for dimensionality other than 2!'

font_size = 12
linewidth = 2.0
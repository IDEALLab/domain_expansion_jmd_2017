"""
A Gaussian Process Classifier interface for the active learning module

Author(s): Wei Chen (wchen459@umd.edu)
"""
import logging
LOGGER = logging.getLogger(__name__)

import numpy as np
from gpc import GPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor

from libact.base.interfaces import ContinuousModel


class GPC(ContinuousModel):

    """ Gaussian Process Classifier """

    def __init__(self, *args, **kwargs):
#        self.model = pyGPs.gp.GPC(*args, **kwargs)
        self.model = GPClassifier(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        self.model.fit(*(dataset.format_sklearn() + args), **kwargs)
        return self.model

    def predict(self, feature, *args, **kwargs):
        feature = np.array(feature)
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        feature = np.array(feature)
        dvalue = self.model.predict_proba(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue
        
    def predict_mean_var(self, feature, *args, **kwargs):
        feature = np.array(feature)
        return self.model.predict_proba(feature, get_var=True, *args, **kwargs)
        
    def get_kernel(self):
        return self.model.kernel_
        
    def get_log_marginal_likelihood(self):
        return self.model.log_marginal_likelihood_value_
        
        
class GPR(ContinuousModel):
    
    """ Gaussian Process Regression """
    
    def __init__(self, *args, **kwargs):
        self.model = GaussianProcessRegressor(*args, **kwargs)
        
    def train(self, dataset, *args, **kwargs):
        self.model.fit(*(dataset.format_sklearn() + args), **kwargs)
        return self.model

    def predict(self, feature, *args, **kwargs):
        feature = np.array(feature)
        f = self.model.predict(feature, *args, **kwargs)-np.finfo(float).eps
        return np.sign(f)

    def predict_real(self, feature, *args, **kwargs):
        feature = np.array(feature)
        dvalue = self.model.predict(feature, *args, **kwargs)
        return np.vstack((-dvalue, dvalue)).T
        
    def predict_mean_var(self, feature, sigma_n=.01, *args, **kwargs):
        feature = np.array(feature)
        t_mean, y_std = self.model.predict(feature, return_std=True, *args, **kwargs)
        t_var = y_std**2 + sigma_n**2 # Gaussian noise model
        return t_mean.flatten(), t_var.flatten()
        
    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
        
    def get_kernel(self):
        return self.model.kernel_
        
    def get_log_marginal_likelihood(self):
        return self.model.log_marginal_likelihood_value_
        
"""
Query strategies for active learning

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import norm
from sklearn.metrics.pairwise import pairwise_distances

from libact.base.interfaces import QueryStrategy, ContinuousModel
from libact.utils import inherit_docstring_from, seed_random_state, zip
from libact.query_strategies import UncertaintySampling

        
class UncertSampling(UncertaintySampling):

    """ Uncertainty Sampling """
    
    def make_query(self):
        dataset = self.dataset
        self.model.train(dataset)

        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())

        if self.method == 'lc':  # least confident
            ask_id = np.argmin(
                np.max(self.model.predict_real(X_pool), axis=1)
            )
#            ask_id = np.argmin(np.abs(self.model.predict_mean_var(X_pool)[0]))

        elif self.method == 'sm':  # smallest margin
            dvalue = self.model.predict_real(X_pool)

            if np.shape(dvalue)[1] > 2:
                # Find 2 largest decision values
                dvalue = -(np.partition(-dvalue, 2, axis=1)[:, :2])

            margin = np.abs(dvalue[:, 0] - dvalue[:, 1])
            ask_id = np.argmin(margin)
            
#        print np.max(self.model.predict_real(X_pool), axis=1)[ask_id]

        return unlabeled_entry_ids[ask_id], self.model


class EpsilonMarginSampling(QueryStrategy):
    
    """ Probabilistic uncertainty sampling """
    
    def __init__(self, *args, **kwargs):
        super(EpsilonMarginSampling, self).__init__(*args, **kwargs)

        self.model = kwargs.pop('model', None)
        self.margin = kwargs.pop('margin', .7)
        self.eta = kwargs.pop('eta', .8)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        if not isinstance(self.model, ContinuousModel):
            raise TypeError(
                "model has to be a ContinuousModel"
            )
        self.model.train(self.dataset)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self, center):
        dataset = self.dataset
        margin = self.margin
        self.model.train(dataset)

        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        mean, var = self.model.predict_mean_var(X_pool)
        l = self.model.get_kernel().get_params()['length_scale']

        # Probability of being in the other class with some certainty
        prob = norm.cdf(-(np.abs(mean)+margin)/np.sqrt(var))
#        prob = np.exp(-(np.abs(mean)+margin)/np.sqrt(var))
        
        t = norm.cdf(-margin) * self.eta  # higher -> explore faster
        dist = np.linalg.norm(X_pool-center, axis=1)
        score = dist + 1e20*(prob < t)
        ask_id = np.argmin(score)
        
#        print score[ask_id]

        return unlabeled_entry_ids[ask_id], self.model



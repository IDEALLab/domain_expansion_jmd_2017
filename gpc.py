"""
Gaussian processes classification that computes the variance

Author(s): Wei Chen (wchen459@umd.edu)

References:
-----------
http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier
"""


import numpy as np
from scipy.linalg import solve
from scipy.special import erf

from sklearn.utils.validation import check_is_fitted
from sklearn.gaussian_process.gpc import _BinaryGaussianProcessClassifierLaplace
from sklearn.base import ClassifierMixin


# Values required for approximating the logistic sigmoid by
# error functions. coefs are obtained via:
# x = np.array([0, 0.6, 2, 3.5, 4.5, np.inf])
# b = logistic(x)
# A = (erf(np.dot(x, self.lambdas)) + 1) / 2
# coefs = lstsq(A, b)[0]
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
COEFS = np.array([-1854.8214151, 3516.89893646, 221.29346712,
                  128.12323805, -2010.49422654])[:, np.newaxis]
                  

class GPClassifier(_BinaryGaussianProcessClassifierLaplace, ClassifierMixin):
    
    def predict_proba(self, X, get_var=False):
        """Return probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
        Returns
        -------
        C : array-like, shape = (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute ``classes_``.
        """
        check_is_fitted(self, ["X_train_", "y_train_", "pi_", "W_sr_", "L_"])

        # Based on Algorithm 3.2 of GPML
        K_star = self.kernel_(self.X_train_, X)  # K_star =k(x_star)
        f_star = K_star.T.dot(self.y_train_ - self.pi_)  # Line 4
        v = solve(self.L_, self.W_sr_[:, np.newaxis] * K_star)  # Line 5
        
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        var_f_star = self.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)
                
        if get_var:
            return f_star, var_f_star
            
        # Line 7:
        # Approximate \int log(z) * N(z | f_star, var_f_star)
        # Approximation is due to Williams & Barber, "Bayesian Classification
        # with Gaussian Processes", Appendix A: Approximate the logistic
        # sigmoid by a linear combination of 5 error functions.
        # For information on how this integral can be computed see
        # blitiri.blogspot.de/2012/11/gaussian-integral-of-error-function.html
        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * f_star
        integrals = np.sqrt(np.pi / alpha) \
            * erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS**2))) \
            / (2 * np.sqrt(var_f_star * 2 * np.pi))
        pi_star = (COEFS * integrals).sum(axis=0) + .5 * COEFS.sum()
        
        return np.vstack((1 - pi_star, pi_star)).T

        

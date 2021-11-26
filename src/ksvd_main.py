import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram
#from typing import List, Any, Dict, Tuple


class ApproxKSVD:
    def __init__(self, num_topics: int, num_words: int, max_iters: int=10, err_tol: float=1e-6) -> None:
        """
        @param num_topics: Number of topics (i.e., atoms or dictionary elements)
        @param num_words: how many atoms each word can load onto
        @param max_iters: Maximum number of iterations
        @param err_tol: Error tolerance
        """
        self.components_ = None
        self.max_iters = max_iters
        self.err_tol = err_tol
        self.num_topics = num_topics
        self.num_words = num_words

    def _update_dict(self, X, D, weights):
        """
        Updates dictionary and weights
        @param X
        @param D
        @param weights
        @return Dictionary, Weight
        """
        for j in range(self.num_topics):
            I = weights[:, j] > 0
            if np.sum(I) == 0:
                continue

            D[j, :] = 0
            w = weights[I, j].T
            r = X[I, :] - weights[I, :].dot(D)
            d = r.T.dot(w)
            d /= np.linalg.norm(d)
            w = r.dot(d)
            D[j, :] = d
            weights[I, j] = w.T
        return D, weights

    def _initialize(self, X):
        """
        Apply svd on data set
        @param X:
        @return ApproxKSVD object
        """
        Ntopics = self.num_topics
        if min(X.shape) < Ntopics:
            D = np.random.randn(Ntopics, X.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(X, k=Ntopics)
            D = np.dot(np.diag(s), vt)
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _transform(self, D, X):
        """
        Sparse Coding task - find best coefficients of dictionary 
        @param D:
        @param X:
        @return weights
        """
        return orthogonal_mp_gram(
            Gram=D.dot(D.T), Xy=D.dot(X.T), n_nonzero_coefs=self.num_words).T

    def fit(self, X) -> ApproxKSVD:
        """
        Apply svd on data set 
        @param X
        @return ApproxKSVD object
        """
        D = self._initialize(X)
        for i in range(self.max_iters):
            w = self._transform(D, X)
            e = np.linalg.norm(X - w.dot(D))
            if e < self.err_tol:
                break
            D, w = self._update_dict(X, D, w)

        self.components_ = D
        return self

    def transform(self, X):
        """
        Apply svd on data set 
        @param X
        @return ApproxKSVD object
        """
        return self._transform(self.components_, X)

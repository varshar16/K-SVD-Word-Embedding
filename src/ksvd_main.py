import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram
from typing import Any


class ApproxKSVD:
    def __init__(self, num_topics: int, num_words: int, iters: int=10,
                 err_tol:float=1e-6) -> None:
        """
        @param num_topics: Number of topics (i.e., atoms or dictionary elements)
        @param num_words: how many atoms each word can load onto
        @param iters: Number of iterations
        @param err_tol: Error tolerance
        """
        self.iters = iters
        self.err_tol = err_tol
        self.num_topics = num_topics
        self.num_words = num_words

    def _update_dict(self, X, D, weights):
        """
        Updates dictionary and weights
        @param X: Word Vectors
        @param D: Dictionary of discourse atoms
        @param weights: contain the loadings of each word onto the discourse atoms
        @return Dictionary, Weight
        """
        for j in range(self.num_topics):
            # Fetch indices of signals in X where non-zero values of d_j are represented.
            I = weights[:, j] > 0
            if np.sum(I) == 0:
                continue
            # Set jth row and all the columns to zeroes
            D[j, :] = 0
            w = weights[I, j].T
            # residual
            r = X[I, :] - weights[I, :].dot(D)
            d = r.T.dot(w)
            d /= np.linalg.norm(d)
            w = r.dot(d)
            D[j, :] = d
            weights[I, j] = w.T
        return D, weights

    def _initialize(self, X):
        """
        Intializes dictionary from given matrix
        @param X: Word Vectors
        @return Dictionary of discourse atoms
        """
        Ntopics = self.num_topics
        # Reduce dimensions of X if it is greater than required number of topics
        if min(X.shape) < Ntopics:
            D = np.random.randn(Ntopics, X.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(X, k=Ntopics)
            D = np.dot(np.diag(s), vt)
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _transform(self, D, X):
        """
        Sparse Coding using Orthogonal Matching Pursuit method to find best
        coefficients of dictionary
        @param D: Dictionary of discourse atoms
        @param X: Word Vectors
        @return weights of each word onto discourse atoms
        """
        return orthogonal_mp_gram(
            Gram=D.dot(D.T), Xy=D.dot(X.T), n_nonzero_coefs=self.num_words).T

    def fit(self, X):
        """
        Apply approximate k-svd on data set, to get best dictionary and
        coefficients
        @param X: Word Vectors
        @return dictionary and weights
        """
        D = self._initialize(X)
        for i in range(self.iters):
            # Sparse Coding
            w = self._transform(D, X)
            # check if the distances are less than target sparsity
            e = np.linalg.norm(X - w.dot(D))
            if e < self.err_tol:
                break
            # Update dictionary and weights
            D, w = self._update_dict(X, D, w)

        return D, w

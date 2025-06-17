import pandas as pd
import numpy as np
from lingam.base import _BaseLiNGAM
from lingam.hsic import hsic_test_gamma

class HSICDirectLiNGAM(_BaseLiNGAM):
    """
    Implementation of DirectLiNGAM using HSIC as the independence measure.

    This version uses a greedy search to find the causal order by minimizing
    the HSIC statistic at each step, making it a direct HSIC-based analogue
    to the standard DirectLiNGAM algorithm.
    """

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)

    def _residual(self, dep_var, indep_var):
        """Calculate the residual of a linear regression."""
        return dep_var - (np.cov(dep_var, indep_var)[0, 1] / np.var(indep_var)) * indep_var

    def _get_hsic(self, vec1, vec2):
        """Calculate the HSIC statistic between two vectors/matrices."""
        # Ensure inputs are 2D arrays for hsic_test_gamma
        if vec1.ndim == 1:
            vec1 = vec1.reshape(-1, 1)
        if vec2.ndim == 1:
            vec2 = vec2.reshape(-1, 1)
            
        hsic_stat, _ = hsic_test_gamma(vec1, vec2)
        return hsic_stat

    def _estimate_order(self, X):
        """Greedy search for causal order based on minimizing HSIC."""
        current_vars = list(range(X.shape[1]))
        causal_order = []

        while len(current_vars) > 0:
            min_hsic = np.inf
            best_candidate = -1

            for candidate_root in current_vars:
                # The other variables are considered the "effects"
                effects = [v for v in current_vars if v != candidate_root]
                
                if not effects:
                    # If it's the last variable, its HSIC is 0
                    hsic_val = 0
                else:
                    # The independence measure is between the candidate root and
                    # the joint set of its effects. In a simple LiNGAM model,
                    # we can check against the effects directly.
                    hsic_val = self._get_hsic(X[:, [candidate_root]], X[:, effects])

                if hsic_val < min_hsic:
                    min_hsic = hsic_val
                    best_candidate = candidate_root
            
            # Add the best candidate to the causal order and remove it from the set
            causal_order.append(best_candidate)
            current_vars.remove(best_candidate)
            
        return causal_order

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self._column_names = X.columns
            X = X.values
        
        # Estimate the causal order using the greedy HSIC method
        self._causal_order = self._estimate_order(X)
        
        # Estimate the adjacency matrix based on the found order
        self._estimate_adjacency_matrix(X)
        return self


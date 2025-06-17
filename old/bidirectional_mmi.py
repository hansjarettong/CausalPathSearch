import os
import argparse
import pandas as pd
import numpy as np
from itertools import combinations
import warnings
import time

# --- Import Causal Discovery Models ---
from lingam.base import _BaseLiNGAM
from lingam.hsic import hsic_test_gamma
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

class BidirectionalMMI(_BaseLiNGAM):
    """
    Implements a true bidirectional, best-first search for causal order discovery.

    - The search expands the most promising node from either the forward or backward
      direction based on the minimum accumulated path cost (HSIC statistic).
    - Forward search (top-down) uses the LiNGAM principle.
    - Backward search (bottom-up) uses the RESIT principle.
    """

    def __init__(self, regressor_bwd, random_state=None):
        super().__init__(random_state)
        if regressor_bwd is None:
            raise ValueError("A backward (nonlinear) regressor must be provided.")
        self._regressor_bwd = regressor_bwd

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self._column_names = X.columns
            X = X.values

        self._causal_order = self._estimate_order_bidirectional(X)
        self._estimate_adjacency_matrix(X) # From base class
        return self

    def _get_independence_measure(self, vec1, vec2):
        """
        Calculates the HSIC statistic. This is the core "cost" function for our path search.
        """
        if vec1.ndim == 1: vec1 = vec1.reshape(-1, 1)
        if vec2.ndim == 1: vec2 = vec2.reshape(-1, 1)
        hsic_stat, _ = hsic_test_gamma(vec1, vec2)
        return hsic_stat
        
    def _linear_residual(self, dep_var, indep_var):
        """The residual when dep_var is regressed on indep_var using the covariance formula."""
        return dep_var - (np.cov(dep_var, indep_var)[0, 1] / np.var(indep_var)) * indep_var

    def _estimate_order_bidirectional(self, X):
        """
        Performs a bidirectional, best-first shortest-path search.
        """
        n_features = X.shape[1]
        all_vars = frozenset(range(n_features))

        # --- Forward Search (Top-to-Bottom, LiNGAM principle) ---
        fwd_open = {frozenset(): 0} # {node: distance}
        fwd_path = {frozenset(): []} # {node: path_list}
        fwd_closed = set()

        # --- Backward Search (Bottom-to-Top, RESIT principle) ---
        bwd_open = {frozenset(): 0} # {node: distance}
        bwd_path = {frozenset(): []} # {node: path_list}
        bwd_closed = set()

        # --- Main Bidirectional Loop ---
        while fwd_open and bwd_open:
            
            # --- Best-First Decision: Choose which direction to expand ---
            fwd_best_dist = min(fwd_open.values())
            bwd_best_dist = min(bwd_open.values())

            if fwd_best_dist <= bwd_best_dist:
                # --- Expand Forward (LiNGAM) ---
                current_node = min(fwd_open, key=fwd_open.get)
                current_dist = fwd_open.pop(current_node)
                current_path = fwd_path[current_node]
                fwd_closed.add(current_node)

                remaining_vars = all_vars - current_node
                for i in remaining_vars:
                    successor = current_node | {i}
                    effects = remaining_vars - {i}
                    
                    cost = 0
                    if effects:
                        residuals = [self._linear_residual(X[:, j], X[:, i]) for j in effects]
                        cost = self._get_independence_measure(X[:, i], np.array(residuals).T)

                    new_dist = current_dist + cost
                    if successor not in fwd_open or new_dist < fwd_open[successor]:
                        fwd_open[successor] = new_dist
                        fwd_path[successor] = current_path + [i]
            else:
                # --- Expand Backward (RESIT) ---
                current_node = min(bwd_open, key=bwd_open.get)
                current_dist = bwd_open.pop(current_node)
                current_path = bwd_path[current_node]
                bwd_closed.add(current_node)

                remaining_vars = all_vars - current_node
                for i in remaining_vars:
                    successor = current_node | {i}
                    predictors = list(remaining_vars - {i})

                    cost = 0
                    if predictors:
                        self._regressor_bwd.fit(X[:, predictors], X[:, i])
                        residual = X[:, i] - self._regressor_bwd.predict(X[:, predictors])
                        cost = self._get_independence_measure(residual, X[:, predictors])

                    new_dist = current_dist + cost
                    if successor not in bwd_open or new_dist < bwd_open[successor]:
                        bwd_open[successor] = new_dist
                        bwd_path[successor] = current_path + [i]

            # --- Check for meeting point ---
            for fwd_node, fwd_p in fwd_path.items():
                complement_vars = all_vars - fwd_node
                if complement_vars in bwd_path:
                    bwd_p = bwd_path[complement_vars]
                    if len(fwd_p) + len(bwd_p) == n_features:
                        print("--- Search frontiers met! ---")
                        return fwd_p + bwd_p[::-1]

        # Fallback in case the search doesn't meet
        print("Warning: Bidirectional search did not meet. Falling back to forward search.")
        final_node = min({k:v for k,v in fwd_open.items() if len(k) == n_features}, key=fwd_open.get)
        return fwd_path[final_node]


# --- Example Usage ---
if __name__ == '__main__':
    from data_handler_v3 import generate_synthetic_dataset

    print("Generating sample nonlinear data with confounders...")
    dataset = generate_synthetic_dataset(
        n_nodes=6, 
        n_samples=1000, 
        graph_density=1.5, 
        n_confounders=2, 
        is_linear=False
    )
    X_df = dataset['X']
    true_order = dataset['causal_order']
    print(f"True causal order: {true_order}")

    print("\nRunning True Bidirectional Search (LiNGAM-Fwd, RESIT-Bwd)...")
    model = BidirectionalMMI(
        regressor_bwd=GradientBoostingRegressor(n_estimators=10)
    )
    model.fit(X_df)
    print(f"Predicted Bidirectional order: {model.causal_order_}")

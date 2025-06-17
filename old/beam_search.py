import pandas as pd
import numpy as np
import heapq
from typing import List

# Import base classes from the user's project structure
from lingam.base import _BaseLiNGAM
from lingam.hsic import hsic_test_gamma
from lingam_mmi import HSIC_LiNGAM_MMI
from resit_mmi import HSIC_RESIT_MMI
from bidirectional_mmi import BidirectionalMMI
from sklearn.ensemble import GradientBoostingRegressor


class _BaseBeamSearch:
    """
    Base class providing the core beam search logic.
    This is not intended to be used directly.
    """
    def __init__(self, beam_width=5, *args, **kwargs):
        # The beam_width (k) determines how many best paths to keep at each step.
        if beam_width < 1:
            raise ValueError("Beam width must be at least 1.")
        self.beam_width = beam_width
        super().__init__(*args, **kwargs)

    def _beam_search_estimate_order(self, X, cost_function):
        """
        Performs a beam search to find the best causal order.

        Args:
            X (np.ndarray): The input data.
            cost_function (function): A function that takes (current_X, candidate_var, remaining_vars)
                                      and returns the cost for that step.
        
        Returns:
            list: The estimated causal order.
        """
        n_features = X.shape[1]
        
        # The beam stores tuples of (total_cost, path_list)
        # A path_list is a list of variables in the order they were chosen.
        beam = [(0, [])]
        
        for _ in range(n_features):
            candidates = []
            for cost, path in beam:
                remaining_vars = [i for i in range(n_features) if i not in path]
                
                if not remaining_vars:
                    # If path is complete, add it to candidates to be considered for the final beam
                    heapq.heappush(candidates, (cost, path))
                    continue

                for var in remaining_vars:
                    # Calculate the cost for selecting 'var' at this step
                    step_cost = cost_function(X, path, var)
                    new_cost = cost + step_cost
                    new_path = path + [var]
                    heapq.heappush(candidates, (new_cost, new_path))
            
            # Prune the candidates to keep only the top `beam_width` paths
            beam = heapq.nsmallest(self.beam_width, candidates)

        # After the loop, the best path is the one with the minimum cost
        if not beam:
            raise RuntimeError("Beam search finished with an empty beam. This should not happen.")
            
        best_cost, best_path = beam[0]
        return best_path


class HSIC_LiNGAM_Beam(HSIC_LiNGAM_MMI):
    """
    HSIC_LiNGAM using Beam Search instead of Dijkstra's.
    """
    def __init__(self, beam_width=5, random_state=None, known_ordering=None):
        super().__init__(random_state=random_state, known_ordering=known_ordering)
        self.beam_width = beam_width
        
    def _get_lingam_cost(self, X, path, candidate_var):
        """Cost function for LiNGAM (top-down)."""
        # Calculate residuals based on the current path
        current_X = X
        for var in path:
            effects = [j for j in range(X.shape[1]) if j not in path and j != var]
            residuals = np.array([self._residual(current_X[:, j], current_X[:, var]) for j in effects]).T
            current_X = np.hstack([current_X[:, path + [var]], residuals])

        # Cost is HSIC between candidate and the residuals of its effects
        all_vars = list(range(X.shape[1]))
        effects = [v for v in all_vars if v not in path and v != candidate_var]
        
        if not effects:
            return 0

        residuals = np.array([self._residual(current_X[:, j], current_X[:, candidate_var]) for j in effects]).T
        hsic_stat, _ = hsic_test_gamma(current_X[:, [candidate_var]], residuals)
        return hsic_stat
        
    def _estimate_order(self, X):
        return _BaseBeamSearch._beam_search_estimate_order(self, X, self._get_lingam_cost)


class HSIC_RESIT_Beam(HSIC_RESIT_MMI):
    """
    HSIC_RESIT using Beam Search instead of Dijkstra's.
    """
    def __init__(self, regressor, beam_width=5, random_state=None, alpha=0.01):
        super().__init__(regressor=regressor, random_state=random_state, alpha=alpha)
        self.beam_width = beam_width

    def _get_resit_cost(self, X, path, candidate_var):
        """Cost function for RESIT (bottom-up)."""
        all_vars = list(range(X.shape[1]))
        # In RESIT, the "path" represents the set of effects (children) already chosen.
        # The 'candidate_var' is the potential effect we are choosing now.
        # The predictors are the remaining variables.
        predictors = [v for v in all_vars if v not in path and v != candidate_var]

        if not predictors:
            return 0

        self._reg.fit(X[:, predictors], X[:, candidate_var])
        residual = X[:, candidate_var] - self._reg.predict(X[:, predictors])
        
        hsic_stat, _ = hsic_test_gamma(residual.reshape(-1, 1), X[:, predictors])
        return hsic_stat

    def _estimate_order(self, X):
        # RESIT search is bottom-up (finding effects first), so we reverse the final path.
        bottom_up_order = _BaseBeamSearch._beam_search_estimate_order(self, X, self._get_resit_cost)
        return bottom_up_order[::-1]


class BidirectionalBeamSearch(BidirectionalMMI):
    """
    Bidirectional search using beams for both forward and backward searches.
    """
    def __init__(self, regressor_bwd, beam_width=5, random_state=None):
        super().__init__(regressor_bwd=regressor_bwd, random_state=random_state)
        self.beam_width = beam_width

    def _estimate_order_bidirectional(self, X):
        """
        Performs a bidirectional search using two beams.
        """
        n_features = X.shape[1]
        all_vars = frozenset(range(n_features))

        # Forward (LiNGAM) beam
        fwd_beam = [(0, [])]  # List of (cost, path)
        fwd_visited = {frozenset(): 0}

        # Backward (RESIT) beam
        bwd_beam = [(0, [])]  # List of (cost, path)
        bwd_visited = {frozenset(): 0}

        # LiNGAM and RESIT cost functions
        lingam_beam_estimator = HSIC_LiNGAM_Beam(beam_width=self.beam_width)
        resit_beam_estimator = HSIC_RESIT_Beam(regressor=self._regressor_bwd, beam_width=self.beam_width)

        for i in range(n_features // 2 + 1):
            # --- Expand Forward ---
            fwd_candidates = []
            for cost, path in fwd_beam:
                remaining_vars = [v for v in range(n_features) if v not in path]
                for var in remaining_vars:
                    step_cost = lingam_beam_estimator._get_lingam_cost(X, path, var)
                    new_path = path + [var]
                    new_cost = cost + step_cost
                    
                    path_set = frozenset(new_path)
                    if path_set not in fwd_visited or new_cost < fwd_visited[path_set]:
                        fwd_visited[path_set] = new_cost
                        heapq.heappush(fwd_candidates, (new_cost, new_path))
            fwd_beam = heapq.nsmallest(self.beam_width, fwd_candidates)

            # --- Expand Backward ---
            bwd_candidates = []
            for cost, path in bwd_beam:
                remaining_vars = [v for v in range(n_features) if v not in path]
                for var in remaining_vars:
                    step_cost = resit_beam_estimator._get_resit_cost(X, path, var)
                    new_path = path + [var]
                    new_cost = cost + step_cost

                    path_set = frozenset(new_path)
                    if path_set not in bwd_visited or new_cost < bwd_visited[path_set]:
                        bwd_visited[path_set] = new_cost
                        heapq.heappush(bwd_candidates, (new_cost, new_path))
            bwd_beam = heapq.nsmallest(self.beam_width, bwd_candidates)

            # --- Check for meeting point ---
            for fwd_cost, fwd_path in fwd_beam:
                fwd_path_set = frozenset(fwd_path)
                for bwd_cost, bwd_path in bwd_beam:
                    bwd_path_set = frozenset(bwd_path)
                    if fwd_path_set.isdisjoint(bwd_path_set) and len(fwd_path) + len(bwd_path) == n_features:
                        # Found a full path where the two searches meet
                        print("--- Bidirectional beams met! ---")
                        return fwd_path + bwd_path[::-1]

        # Fallback if no meeting point is found (should be rare)
        print("Warning: Bidirectional search did not meet. Falling back to the best forward path.")
        final_forward_search = HSIC_LiNGAM_Beam(beam_width=self.beam_width)
        return final_forward_search._estimate_order(X)


# --- Example Usage ---
if __name__ == '__main__':
    from data_handler import generate_synthetic_dataset

    print("Generating sample nonlinear data with confounders...")
    dataset = generate_synthetic_dataset(
        n_nodes=6,
        n_samples=500,
        graph_density=1.5,
        n_confounders=1,
        is_linear=False,
        noise_type='gaussian'
    )
    X_df = dataset['X']
    true_order = dataset['causal_order']
    print(f"True causal order: {true_order}")

    BEAM_WIDTH = 5
    print(f"\n--- Running Beam Search Models (k={BEAM_WIDTH}) ---")

    # --- HSIC LiNGAM with Beam Search ---
    print("\nRunning HSIC LiNGAM (Beam Search)...")
    model_lingam_beam = HSIC_LiNGAM_Beam(beam_width=BEAM_WIDTH)
    model_lingam_beam.fit(X_df)
    print(f"Predicted LiNGAM Beam order: {model_lingam_beam.causal_order_}")

    # --- HSIC RESIT with Beam Search ---
    print("\nRunning HSIC RESIT (Beam Search)...")
    model_resit_beam = HSIC_RESIT_Beam(
        regressor=GradientBoostingRegressor(n_estimators=10),
        beam_width=BEAM_WIDTH
    )
    model_resit_beam.fit(X_df)
    print(f"Predicted RESIT Beam order: {model_resit_beam.causal_order_}")

    # --- Bidirectional with Beam Search ---
    print("\nRunning Bidirectional (Beam Search)...")
    model_bidir_beam = BidirectionalBeamSearch(
        regressor_bwd=GradientBoostingRegressor(n_estimators=10),
        beam_width=BEAM_WIDTH
    )
    model_bidir_beam.fit(X_df)
    print(f"Predicted Bidirectional Beam order: {model_bidir_beam.causal_order_}")

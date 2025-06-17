import numpy as np
import pandas as pd
from typing import List
from lingam import RESIT
from lingam.hsic import hsic_test_gamma
from lingam.base import _BaseLiNGAM


class _BaseRESIT_MMI(_BaseLiNGAM): # Inherit from _BaseLiNGAM for adjacency matrix estimation
    def __init__(self, regressor, random_state=None, alpha=0.01):
        super().__init__(random_state)
        if regressor is None:
            raise ValueError("Specify regression model in 'regressor'.")
        else:
            if not (hasattr(regressor, "fit") and hasattr(regressor, "predict")):
                raise ValueError("'regressor' has no fit or predict method.")
        self._reg = regressor
        self._alpha = alpha


    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Step 1: Determine topological order using MMI search
        self._causal_order = self._estimate_order(X)

        # Step 2: Estimate the adjacency matrix by pruning edges
        self._estimate_adjacency_matrix(X)
        return self

    def _estimate_adjacency_matrix(self, X):
        """
        Estimates the adjacency matrix from the causal order by removing
        superfluous edges. This is the second step of the RESIT algorithm.
        """
        n_features = X.shape[1]
        
        # Assume a fully connected graph consistent with the causal order
        B = np.zeros((n_features, n_features), dtype=float)
        for i in range(1, n_features):
            k = self._causal_order[i]
            parents = self._causal_order[:i]
            
            # Initially, connect all possible parents
            current_parents = list(parents)
            
            # Iteratively remove parents if they are independent given the others
            for parent_to_test in parents:
                # Predictors are the other parents
                predictors = [p for p in current_parents if p != parent_to_test]
                
                if not predictors:
                    continue

                # Regress child on the other parents
                self._reg.fit(X[:, predictors], X[:, k])
                residual = X[:, k] - self._reg.predict(X[:, predictors])
                
                # If child's residual is independent of the parent_to_test,
                # then the edge is superfluous.
                _, p_val = hsic_test_gamma(residual.reshape(-1, 1), X[:, [parent_to_test]])
                if p_val > self._alpha:
                    current_parents.remove(parent_to_test)

            # Set the final edges in the adjacency matrix
            for p in current_parents:
                # For RESIT, we just mark the connection, not estimate a coefficient
                B[k, p] = 1.0

        self._adjacency_matrix = B
        return self


    def _estimate_order(self, X):
        # using Dijkstra with lazy evaluation of MI
        n_features = X.shape[1]
        start_node = frozenset(range(n_features))
        goal_node = frozenset({})

        OPEN = {start_node}
        distance = {start_node: 0}
        path = {start_node: [start_node]}

        self._edges_computed = 0
        while OPEN:
            # Refer to LiNGAM-MMI paper by Suzuki, Algorithm 1
            # 1. Move the node v with the smallest distance from OPEN to CLOSED
            smallest_node = min(OPEN, key=distance.get)
            OPEN -= {smallest_node}

            # 2. Join the successors of v to OPEN
            successors = self._get_successors(smallest_node)
            OPEN.update(successors)

            # 3. If goal_node in OPEN, append goal node to path(v) -- this is the shortest path. Terminate
            if goal_node in OPEN:
                # The path is from root to leaf, so we need to reverse it to get cause-to-effect order
                full_path = path[smallest_node] + [goal_node]
                return self._path2order(full_path)[::-1] # Reverse the order for RESIT's bottom-up logic


            # 4. Evaluate the mutual information of the successors
            for successor in successors:
                # This logic is for a bottom-up (RESIT) search
                # The selected feature is the one *removed* from the set
                selected_feature_idx = list(smallest_node - successor)[0]
                
                # Predictors are the remaining variables
                predictors = list(successor)
                
                cost = 0
                if predictors:
                    self._reg.fit(X[:, predictors], X[:, selected_feature_idx])
                    residual = X[:, selected_feature_idx] - self._reg.predict(X[:, predictors])
                    cost = self._get_mutual_info(residual, X[:, predictors])
                
                self._edges_computed += 1

                if successor not in distance or (
                    successor in distance
                    and distance[smallest_node] + cost < distance[successor]
                ):
                    distance[successor] = distance[smallest_node] + cost
                    path[successor] = path[smallest_node] + [successor]

    def _get_successors(self, node: frozenset):
        return {node - {i} for i in node}

    def _path2order(self, path: List[frozenset]):
        path_len = len(path)
        return [list(path[i] - path[i + 1])[0] for i in range(path_len - 1)]

    def _get_mutual_info(self, residual, predictors):
        raise NotImplementedError


class HSIC_RESIT_MMI(_BaseRESIT_MMI):
    def __init__(self, regressor, use_pval: bool = False, random_state=None, bw_method="mdbs", alpha=0.01):
        self.bw_method = bw_method
        self.use_pval = use_pval
        super().__init__(regressor, random_state, alpha)

    def _get_mutual_info(self, residual, predictors):
        hsic_stat, pval = hsic_test_gamma(residual.reshape(-1, 1), np.array(predictors), bw_method=self.bw_method)
        return 1-pval if self.use_pval else hsic_stat

# Note: The HybridRESIT class has been removed for clarity as it's not used in the main experiment runner.
# It can be added back if needed, but it would also need the two-step fit process.

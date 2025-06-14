import numpy as np
import pandas as pd
from typing import List
from lingam import RESIT
from lingam.hsic import hsic_test_gamma


class _BaseRESIT_MMI(RESIT):
    def __init__(self, regressor, random_state=None):
        # # Check input parameters
        # if regressor is None:
        #     raise ValueError("Specify regression model in 'regressor'.")
        # else:
        #     if not (hasattr(regressor, "fit") and hasattr(regressor, "predict")):
        #         raise ValueError("'regressor' has no fit or predict method.")

        super().__init__(regressor, random_state)

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        # Determine topological order
        pi = self._estimate_order(X)
        self._causal_order = pi
        # TODO: remove superflous edges
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
                path[smallest_node].append(goal_node)
                return self._path2order(path[smallest_node])

            # 4. Evaluate the mutual information of the successors
            for successor in successors:
                selected_feature_idx = list(smallest_node - successor)[0]
                self._reg.fit(X[:, list(successor)], X[:, selected_feature_idx])
                residual = X[:, selected_feature_idx] - self._reg.predict(
                    X[:, list(successor)]
                )
                # measure mutual information
                mi = self._get_mutual_info(residual, X[:, list(successor)])
                self._edges_computed += 1

                if successor not in distance or (
                    successor in distance
                    and distance[smallest_node] + mi < distance[successor]
                ):
                    distance[successor] = distance[smallest_node] + mi
                    path[successor] = path[smallest_node] + [successor]

    def _get_successors(self, node: frozenset):
        return {node - {i} for i in node}

    def _path2order(self, path: List[frozenset]):
        path_len = len(path)
        return [list(path[i] - path[i + 1])[0] for i in range(path_len - 1)][::-1]

    def _get_mutual_info(self, single_vec, many_vec):
        raise NotImplementedError


class HSIC_RESIT_MMI(_BaseRESIT_MMI):
    def __init__(self, regressor, use_pval: bool = False, random_state=None, bw_method="mdbs"):
        self.bw_method = bw_method
        self.use_pval = use_pval
        super().__init__(regressor, random_state)

    def _get_mutual_info(self, residual, predictors):
        hsic_stat, pval = hsic_test_gamma(np.array(residual), np.array(predictors), bw_method=self.bw_method)
        return 1-pval if self.use_pval else hsic_stat

class HybridRESIT(HSIC_RESIT_MMI):
    """Do RESIT until no further edges can be removed. Then do RESIT-MMI."""
    def __init__(self, regressor, random_state=None, alpha = 0.01):
        super().__init__(regressor, random_state)
        self._alpha = alpha

    def _estimate_order(self, X):
        """Determine topological order (copied from original RESIT code)"""
        S = np.arange(X.shape[1])
        pa = {}
        pi = []
        self._edges_computed = 0
        for _ in range(X.shape[1]):
            if len(S) == 1:
                pa[S[0]] = []
                pi.insert(0, S[0])
                continue

            hsic_stats = []
            hsic_pvals = [] 
            for k in S:
                # Regress Xk on {Xi}
                predictors = [i for i in S if i != k]
                self._reg.fit(X[:, predictors], X[:, k])
                residual = X[:, k] - self._reg.predict(X[:, predictors])
                # Measure dependence between residuals and {Xi}
                hsic_stat, hsic_p = hsic_test_gamma(residual, X[:, predictors])
                self._edges_computed += 1
                hsic_stats.append(hsic_stat)
                hsic_pvals.append(hsic_p)

            # if there are no independent candidates
            if min(hsic_pvals) > self._alpha:
                break

            k = S[np.argmin(hsic_stats)]
            S = S[S != k]
            pa[k] = S.tolist()
            pi.insert(0, k)
        # if it didn't break
        else:
            return pi


        # using Dijkstra with lazy evaluation of MI
        n_features = X.shape[1]
        start_node = frozenset(range(n_features)) - frozenset(pi)
        goal_node = frozenset({})

        OPEN = {start_node}
        distance = {start_node: 0}
        path = {start_node: [start_node]}

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
                path[smallest_node].append(goal_node)
                return self._path2order(path[smallest_node]) + pi

            # 4. Evaluate the mutual information of the successors
            for successor in successors:
                selected_feature_idx = list(smallest_node - successor)[0]
                self._reg.fit(X[:, list(successor)], X[:, selected_feature_idx])
                residual = X[:, selected_feature_idx] - self._reg.predict(
                    X[:, list(successor)]
                )
                # measure mutual information
                mi = self._get_mutual_info(residual, X[:, list(successor)])
                self._edges_computed += 1

                if successor not in distance or (
                    successor in distance
                    and distance[smallest_node] + mi < distance[successor]
                ):
                    distance[successor] = distance[smallest_node] + mi
                    path[successor] = path[smallest_node] + [successor]
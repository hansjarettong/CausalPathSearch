import heapq
import time
import warnings
from itertools import combinations
from multiprocessing import cpu_count

import numpy as np
from lingam.hsic import hsic_test_gamma
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, LassoLarsIC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


class CausalSearch:
    """
    Base class for causal discovery models using a unified search framework.
    """

    def __init__(self, regressor=None, beam_width=float('inf'), independence_measure='statistic', prune=False, alpha=0.01, n_jobs=-1):
        self.regressor = regressor if regressor is not None else LinearRegression()
        self.beam_width = beam_width
        self.independence_measure = independence_measure
        self.prune = prune
        self.alpha = alpha
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.causal_order_ = None
        self.adjacency_matrix_ = None

    def fit(self, X, y=None):
        self.causal_order_, self.adjacency_matrix_ = self._estimate_causal_order(X)
        
        if self.prune:
            self.adjacency_matrix_ = self._prune_edges(X, self.causal_order_, self.adjacency_matrix_)
            
        return self

    def _estimate_causal_order(self, X):
        raise NotImplementedError("This method must be implemented by subclasses.")

    def _beam_search(self, X, start_node_features, is_goal_node, get_next_nodes, get_cost, path_to_order_func):
        start_node = (0, [start_node_features])
        paths_queue = [start_node]
        heapq.heapify(paths_queue)
        
        best_complete_path = (float('inf'), None)

        while paths_queue:
            cost, path = heapq.heappop(paths_queue)
            if cost >= best_complete_path[0]: continue
            current_node_features = path[-1]

            if is_goal_node(current_node_features):
                if cost < best_complete_path[0]:
                    best_complete_path = (cost, path)
                continue

            possible_next_nodes = get_next_nodes(current_node_features)
            for next_feature_set in possible_next_nodes:
                # The full path is now passed to the cost function
                step_cost = get_cost(X, path, current_node_features, next_feature_set)
                if step_cost is None: continue
                new_cost = cost + step_cost
                if new_cost < best_complete_path[0]:
                    new_path = path + [next_feature_set]
                    heapq.heappush(paths_queue, (new_cost, new_path))

            if self.beam_width != float('inf') and len(paths_queue) > self.beam_width:
                paths_queue = heapq.nsmallest(self.beam_width, paths_queue)
                heapq.heapify(paths_queue)

        if best_complete_path[1] is None:
            raise RuntimeError("Causal order could not be found. Search failed.")

        final_order = path_to_order_func(best_complete_path[1])
        adjacency_matrix = self._create_full_dag_from_order(len(X[0]), final_order)
        return final_order, adjacency_matrix
        
    def _prune_edges(self, X, causal_order, initial_adj_matrix):
        n_features = X.shape[1]
        pruned_adj = initial_adj_matrix.copy()
        
        parents_dict = {i: np.where(pruned_adj[i, :] != 0)[0].tolist() for i in range(n_features)}

        for i in range(1, n_features):
            child_node = causal_order[i]
            potential_parents = parents_dict[child_node].copy()

            for parent_to_test in potential_parents:
                other_parents = [p for p in parents_dict[child_node] if p != parent_to_test]
                if not other_parents: continue
                
                reg = self.regressor
                reg.fit(X[:, other_parents], X[:, child_node])
                residual = X[:, child_node] - reg.predict(X[:, other_parents])

                _, p_val = hsic_test_gamma(residual[:, np.newaxis], X[:, other_parents])
                
                if p_val > self.alpha:
                    parents_dict[child_node].remove(parent_to_test)
        
        final_adj = np.zeros_like(pruned_adj)
        for child, parents in parents_dict.items():
            if parents:
                final_adj[child, parents] = 1
        return final_adj

    def _create_full_dag_from_order(self, n_features, causal_order):
        adjacency_matrix = np.zeros((n_features, n_features))
        for i, child in enumerate(causal_order):
            parents = causal_order[:i]
            if parents:
                adjacency_matrix[child, parents] = 1
        return adjacency_matrix

    def _path_to_order(self, path):
        order = []
        for i in range(len(path) - 1):
            diff = path[i] - path[i+1]
            if diff:
                order.append(list(diff)[0])
        return order


class LiNGAM(CausalSearch):
    def __init__(self, regressor=None, beam_width=float('inf'), independence_measure='statistic', n_jobs=-1):
        super().__init__(regressor, beam_width, independence_measure, prune=False, n_jobs=n_jobs)

    def _estimate_causal_order(self, X):
        n_features = X.shape[1]
        start_node_features = frozenset(range(n_features))
        is_goal_node = lambda features: not features
        get_next_nodes = lambda features: [features - {f} for f in features]
            
        causal_order, _ = super()._beam_search(X, start_node_features, is_goal_node, get_next_nodes, self._get_cost, self._path_to_order)
        
        adjacency_matrix = self._estimate_lingam_matrix_with_adaptive_lasso(X, causal_order)
        return causal_order, adjacency_matrix

    def _estimate_lingam_matrix_with_adaptive_lasso(self, X, causal_order):
        B = np.zeros((X.shape[1], X.shape[1]))
        for i in range(1, len(causal_order)):
            target = causal_order[i]
            predictors = causal_order[:i]
            if not predictors:
                continue
            
            B[target, predictors] = self._predict_adaptive_lasso(X, predictors, target)
        return B

    def _predict_adaptive_lasso(self, X, predictors, target, gamma=1.0):
        X_std = StandardScaler().fit_transform(X)
        
        lr = LinearRegression()
        lr.fit(X_std[:, predictors], X_std[:, target])
        weights = np.power(np.abs(lr.coef_), gamma)
        weights = np.where(weights == 0, 1e-10, weights)

        X_weighted_predictors = X_std[:, predictors] * weights
        reg = LassoLarsIC(criterion="bic")
        reg.fit(X_weighted_predictors, X_std[:, target])
        
        selected_features_mask = np.abs(reg.coef_ * weights) > 1e-10
        selected_features = np.array(predictors)[selected_features_mask]
        
        final_coefs = np.zeros(len(predictors))
        if len(selected_features) > 0:
            lr_final = LinearRegression()
            lr_final.fit(X[:, selected_features], X[:, target])
            
            for i, p_idx in enumerate(selected_features):
                original_pos = predictors.index(p_idx)
                final_coefs[original_pos] = lr_final.coef_[i]
        return final_coefs

    def _residual(self, dep_var, indep_var):
        var_indep = np.var(indep_var)
        if var_indep < 1e-10: return dep_var
        return dep_var - (np.cov(dep_var, indep_var)[0, 1] / var_indep) * indep_var

    def _get_cost(self, X, path, current_features, next_features):
        """
        LiNGAM Cost Function. This is path-dependent because it simulates
        the sequential residualization from the original DirectLiNGAM algorithm.
        """
        # Re-create the residualized data matrix based on the path history
        X_temp = X.copy()
        removed_order = self._path_to_order(path)
        
        # Sequentially regress out the effect of already removed variables
        features_in_play = list(range(X.shape[1]))
        for var_k in removed_order:
            features_in_play.remove(var_k)
            for var_i in features_in_play:
                X_temp[:, var_i] = self._residual(X_temp[:, var_i], X_temp[:, var_k])

        # Perform the cost calculation for the current decision using X_temp
        selected_feature = list(current_features - next_features)[0]
        remaining_features = list(next_features)

        if not remaining_features:
            return 0.0

        residuals = np.array([
            self._residual(X_temp[:, target], X_temp[:, selected_feature]) 
            for target in remaining_features
        ]).T
        
        hsic_statistic, p_val = hsic_test_gamma(X_temp[:, [selected_feature]], residuals)
        
        return -p_val if self.independence_measure == 'p_value' else hsic_statistic
    
    def get_binary_adjacency_matrix(self):
        """
        Converts the coefficient-based adjacency matrix to a binary one.
        
        This method checks for non-zero coefficients in a numerically robust way,
        accounting for potential floating-point inaccuracies. An edge is considered
        present if its absolute coefficient is greater than a small tolerance.

        Returns
        -------
        binary_matrix : np.ndarray
            The binary adjacency matrix (0s and 1s).
        """
        if self.adjacency_matrix_ is None:
            raise RuntimeError("The model has not been fitted yet. Call fit() first.")
        
        # Robustly check for non-zero values against a small tolerance (epsilon)
        return (np.abs(self.adjacency_matrix_) > 1e-10).astype(int)


class RESIT(CausalSearch):
    def __init__(self, regressor=None, beam_width=float('inf'), independence_measure='statistic', prune=True, alpha=0.01, n_jobs=-1):
        if regressor is None:
            regressor = GradientBoostingRegressor(n_estimators=20, max_depth=3)
        super().__init__(regressor, beam_width, independence_measure, prune, alpha, n_jobs)

    def _estimate_causal_order(self, X):
        n_features = X.shape[1]
        start_node_features = frozenset(range(n_features))
        is_goal_node = lambda features: not features
        get_next_nodes = lambda features: [features - {f} for f in features]
        path_to_order_reversed = lambda p: self._path_to_order(p)[::-1]
        return self._beam_search(X, start_node_features, is_goal_node, get_next_nodes, self._get_cost, path_to_order_reversed)

    def _get_cost(self, X, path, current_features, next_features):
        # RESIT is stateless with respect to the path, so the path argument is ignored.
        selected_feature = list(current_features - next_features)[0]
        predictors = list(next_features)
        if not predictors: return 0.0
        reg = self.regressor
        reg.fit(X[:, predictors], X[:, selected_feature])
        residuals = X[:, selected_feature] - reg.predict(X[:, predictors])
        hsic_statistic, p_val = hsic_test_gamma(residuals[:, np.newaxis], X[:, predictors])
        return -p_val if self.independence_measure == 'p_value' else hsic_statistic


class BidirectionalMMI(CausalSearch):
    def __init__(self, regressor=None, beam_width=float('inf'), independence_measure='statistic', prune=True, alpha=0.01, n_jobs=-1):
        super().__init__(regressor, beam_width, independence_measure, prune, alpha, n_jobs)
    def _estimate_causal_order(self, X):
        n_features = X.shape[1]
        all_features = frozenset(range(n_features))
        self._forward_searcher = LiNGAM(independence_measure=self.independence_measure)
        self._backward_searcher = RESIT(regressor=self.regressor, independence_measure=self.independence_measure, prune=False)
        forward_q, forward_visited = [(0, [all_features])], {all_features: (0, [all_features])}
        backward_q, backward_visited = [(0, [frozenset()])], {frozenset(): (0, [frozenset()])}
        best_total_cost, meet_node = float('inf'), None
        best_forward_path, best_backward_path = None, None
        while forward_q and backward_q:
            for q, visited, other_visited, searcher, is_forward in [(forward_q, forward_visited, backward_visited, self._forward_searcher, True),
                                                                     (backward_q, backward_visited, forward_visited, self._backward_searcher, False)]:
                if not q: continue
                cost, path = heapq.heappop(q)
                if cost >= best_total_cost: continue
                current_node = path[-1]
                next_nodes = [current_node - {f} for f in current_node] if is_forward else [current_node | {f} for f in all_features - current_node]
                for next_node in next_nodes:
                    # Pass the current path to the cost function
                    step_cost = searcher._get_cost(X, path, current_node if is_forward else next_node, next_node if is_forward else current_node)
                    new_cost = cost + step_cost
                    if new_cost < visited.get(next_node, (float('inf'),))[0]:
                        new_path = path + [next_node]
                        visited[next_node] = (new_cost, new_path)
                        heapq.heappush(q, (new_cost, new_path))
                        if next_node in other_visited:
                            total_cost = new_cost + other_visited[next_node][0]
                            if total_cost < best_total_cost:
                                best_total_cost = total_cost
                                meet_node = next_node
                                best_forward_path = new_path if is_forward else other_visited[next_node][1]
                                best_backward_path = other_visited[next_node][1] if is_forward else new_path
        if meet_node is None:
            warnings.warn("Bidirectional search did not meet. Falling back to forward search.")
            fallback_model = LiNGAM(independence_measure=self.independence_measure)
            return fallback_model.fit(X).causal_order_, fallback_model.adjacency_matrix_
        forward_order = self._path_to_order(best_forward_path)
        backward_path_reversed = best_backward_path[::-1]
        temp_path = [all_features] + [node for node in backward_path_reversed if node != frozenset()]
        backward_order_rev = self._path_to_order(temp_path)[::-1]
        final_order = forward_order + backward_order_rev
        if len(set(final_order)) != n_features:
             warnings.warn("Bidirectional search resulted in an invalid path. Falling back to forward search.")
             fallback_model = LiNGAM(independence_measure=self.independence_measure)
             return fallback_model.fit(X).causal_order_, fallback_model.adjacency_matrix_
        adjacency_matrix = self._create_full_dag_from_order(n_features, final_order)
        return final_order, adjacency_matrix

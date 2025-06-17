import unittest
import numpy as np

# Import from our refactored codebase
from generate_data import DataGenerator
from causal_search import LiNGAM as RefactoredLiNGAM
from causal_search import RESIT as RefactoredRESIT

# Import baseline models from the lingam library
from lingam import DirectLiNGAM as BaselineDirectLiNGAM
from lingam import RESIT as BaselineRESIT
from lingam.hsic import hsic_test_gamma
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# --- Test Configurations ---
DATA_PARAMS = { 'n_nodes': 10, 'n_samples': 800, 'edge_density': 0.5, 'seed': 42 }
ALPHA = 0.01
SEED = 42

class HSICDirectLiNGAM(BaselineDirectLiNGAM):
    """A wrapper to force DirectLiNGAM to use HSIC for ordering."""
    def __init__(self, random_state=None):
        super().__init__(measure='hsic', random_state=random_state)
    
    def _get_measure_score(self, u, v_residuals):
        _, p_val = hsic_test_gamma(u, v_residuals)
        return -p_val

class TestModelEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Generate data once for all tests in this class."""
        print("--- Generating Test Data ---")
        cls.lingam_data = DataGenerator(data_type='linear', noise_dist='uniform', **DATA_PARAMS).generate()
        cls.resit_data = DataGenerator(data_type='nonlinear', noise_dist='gaussian', **DATA_PARAMS).generate()

    def test_lingam_matrix_equivalence(self):
        """Tests if the final matrix from our LiNGAM matches the baseline."""
        print("\n" + "="*20 + " Running LiNGAM Matrix Test " + "="*20)
        X, _, _, _ = self.lingam_data

        # Run our refactored model
        refactored_model = RefactoredLiNGAM(regressor=LinearRegression(), beam_width=1)
        refactored_adj = refactored_model.fit(X).adjacency_matrix_
        print(f"Refactored LiNGAM found order: {refactored_model.causal_order_}")

        # Run the baseline model
        baseline_model = HSICDirectLiNGAM(random_state=SEED)
        baseline_adj = baseline_model.fit(X).adjacency_matrix_
        print(f"Baseline LiNGAM found order:   {baseline_model.causal_order_}")

        np.testing.assert_allclose(
            refactored_adj, baseline_adj, rtol=1e-5, atol=1e-5,
            err_msg="LiNGAM adjacency matrix should match the baseline after Adaptive Lasso pruning."
        )
        print("✅ LiNGAM matrix test passed.")

    def test_resit_matrix_equivalence(self):
        """Tests if the pruned matrix from our RESIT matches the baseline."""
        print("\n" + "="*20 + " Running RESIT Matrix Test " + "="*20)
        X, _, B_true, _ = self.resit_data

        # Run our refactored model
        refactored_regressor = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=SEED)
        refactored_model = RefactoredRESIT(regressor=refactored_regressor, beam_width=1, prune=True, alpha=ALPHA)
        refactored_adj = refactored_model.fit(X).adjacency_matrix_
        print(f"Refactored RESIT found order: {refactored_model.causal_order_}")


        # Run the baseline model
        baseline_regressor = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=SEED)
        baseline_model = BaselineRESIT(regressor=baseline_regressor, alpha=ALPHA)
        baseline_adj = baseline_model.fit(X).adjacency_matrix_
        print(f"Baseline RESIT found order:   {baseline_model.causal_order_}")

        np.testing.assert_array_equal(
            refactored_adj, baseline_adj,
            err_msg="RESIT pruned adjacency matrix should match the baseline."
        )
        print("✅ RESIT pruning test passed.")


if __name__ == '__main__':
    unittest.main(verbosity=2, failfast=False)

import numpy as np
import networkx as nx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings

class DataGenerator:
    """
    A class to generate synthetic data for causal discovery, supporting
    linear/non-linear relationships and hidden confounders.
    """
    def __init__(self, n_nodes, edge_density, n_samples, data_type='linear', 
                 noise_dist='uniform', n_confounders=0, seed=42):
        """
        Initializes the data generator.

        Parameters
        ----------
        n_nodes : int
            Number of observed variables in the causal graph.
        edge_density : float
            The probability of an edge existing between any two nodes.
        n_samples : int
            The number of data points to generate.
        data_type : {'linear', 'nonlinear'}, default='linear'
            The type of causal relationships.
        noise_dist : {'uniform', 'gaussian', 'laplace'}, default='uniform'
            The distribution of the exogenous noise terms.
        n_confounders : int, default=0
            The number of hidden confounders to add to the model.
        seed : int, default=42
            Random seed for reproducibility.
        """
        self.n_nodes = n_nodes
        self.edge_density = edge_density
        self.n_samples = n_samples
        self.data_type = data_type
        self.noise_dist = noise_dist
        self.n_confounders = n_confounders
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        if data_type == 'linear' and noise_dist == 'gaussian':
            warnings.warn("Warning: Linear model with Gaussian noise is unidentifiable. Consider non-Gaussian noise.")
        if data_type == 'nonlinear' and noise_dist != 'gaussian':
            warnings.warn("Warning: Non-linear models typically assume Gaussian noise (Additive Noise Model).")

    def _generate_dag(self):
        """
        Generates a random Directed Acyclic Graph (DAG).
        Returns the adjacency matrix and the causal order.
        """
        B = np.zeros((self.n_nodes, self.n_nodes))
        nodes = np.arange(self.n_nodes)
        self.rng.shuffle(nodes)  # Random causal order
        causal_order = list(nodes)

        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.rng.random() < self.edge_density:
                    # Add edge from node_i to node_j
                    node_i, node_j = causal_order[i], causal_order[j]
                    B[node_j, node_i] = self.rng.uniform(low=0.5, high=2.0) * self.rng.choice([-1, 1])
        
        return B, causal_order

    def _get_noise(self, n_samples, n_vars):
        """Generates noise from the specified distribution."""
        if self.noise_dist == 'uniform':
            return self.rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_vars))
        elif self.noise_dist == 'laplace':
            return self.rng.laplace(loc=0.0, scale=1.0, size=(n_samples, n_vars))
        elif self.noise_dist == 'gaussian':
            return self.rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_vars))
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

    def generate(self):
        """
        Generates the full dataset based on the initialized parameters.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_nodes)
            The generated observed data.
        true_causal_order : list
            The ground truth causal ordering of the nodes.
        B : np.ndarray, shape (n_nodes, n_nodes)
            The ground truth adjacency matrix for the observed variables.
        confounder_data : np.ndarray, shape (n_samples, n_confounders) or None
            The generated hidden confounder data, if any.
        """
        B, causal_order = self._generate_dag()
        X = np.zeros((self.n_samples, self.n_nodes))
        
        # Generate confounders and their effect matrix if specified
        confounder_data = None
        confounder_B = None
        if self.n_confounders > 0:
            # Confounders are exogenous, drawn from a non-Gaussian distribution
            confounder_data = self.rng.uniform(low=-1.0, high=1.0, size=(self.n_samples, self.n_confounders))
            
            # Each confounder affects on average 2 observed nodes
            confounder_B = np.zeros((self.n_nodes, self.n_confounders))
            for i in range(self.n_confounders):
                affected_nodes = self.rng.choice(self.n_nodes, size=2, replace=False)
                confounder_B[affected_nodes, i] = self.rng.uniform(low=0.5, high=2.0, size=2)

        # Generate data following the causal order to ensure propagation
        for j in causal_order:
            # 1. Get parents from observed variables
            parents = np.where(B[j, :] != 0)[0]
            
            # 2. Get parents from confounders
            confounder_parents = []
            if self.n_confounders > 0:
                confounder_parents = np.where(confounder_B[j, :] != 0)[0]

            # 3. Get exogenous noise for the current variable
            noise = self._get_noise(self.n_samples, 1)
            
            # 4. Calculate the value of X_j based on its parents and noise
            total_parent_effect = np.zeros(self.n_samples)

            if self.data_type == 'linear':
                if len(parents) > 0:
                    total_parent_effect += X[:, parents] @ B[j, parents]
                if len(confounder_parents) > 0:
                    total_parent_effect += confounder_data[:, confounder_parents] @ confounder_B[j, confounder_parents]

            elif self.data_type == 'nonlinear':
                all_parents_data = []
                if len(parents) > 0:
                    all_parents_data.append(X[:, parents])
                if len(confounder_parents) > 0:
                    all_parents_data.append(confounder_data[:, confounder_parents])
                
                if all_parents_data:
                    combined_parents = np.hstack(all_parents_data)
                    # The 'C' here now correctly and unambiguously refers to ConstantKernel
                    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                    gp = GaussianProcessRegressor(kernel=kernel, random_state=self.seed)
                    
                    f = gp.sample_y(combined_parents, 1, random_state=self.rng.integers(1e6)).flatten()
                    total_parent_effect += f

            X[:, j] = total_parent_effect + noise.flatten()

        return X, causal_order, B, confounder_data

# Example Usage:
if __name__ == '__main__':
    print("--- Generating Linear Data with a Hidden Confounder ---")
    linear_gen = DataGenerator(
        n_nodes=5, 
        edge_density=0.4, 
        n_samples=1000, 
        data_type='linear',
        noise_dist='uniform',
        n_confounders=1,
        seed=1
    )
    X_linear, order_lin, B_lin, C_lin = linear_gen.generate()
    print("Generated data shape:", X_linear.shape)
    print("True causal order:", order_lin)
    print("Confounder data shape:", C_lin.shape)
    print("\n" + "="*50 + "\n")

    print("--- Generating Non-Linear Data (ANM) without Confounders ---")
    nonlinear_gen = DataGenerator(
        n_nodes=4,
        edge_density=0.5,
        n_samples=500,
        data_type='nonlinear',
        noise_dist='gaussian',
        n_confounders=0,
        seed=2
    )
    X_nonlinear, order_nonlin, B_nonlin, _ = nonlinear_gen.generate()
    print("Generated data shape:", X_nonlinear.shape)
    print("True causal order:", order_nonlin)


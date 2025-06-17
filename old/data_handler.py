import argparse
import numpy as np
import pandas as pd
import os
import networkx as nx
from castle.datasets import DAG
from scipy.stats import t, laplace, uniform, expon, norm
# --- CHANGE: Import GaussianProcessRegressor and its kernel ---
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class NonGaussianDistributions:
    """
    Class to generate a variety of non-Gaussian noise types.
    """
    def __init__(self):
        self.distributions = [
            self.student_3_dof, self.double_exponential, self.uniform,
            self.student_5_dof, self.exponential, self.mixture_of_double_exponentials,
            self.symmetric_mixture_of_two_Gaussians_multimodal,
            self.symmetric_mixture_of_two_Gaussians_transitional,
            self.symmetric_mixture_of_two_Gaussians_unimodal,
            self.nonsymmetric_mixture_of_two_Gaussians_multimodal,
            self.nonsymmetric_mixture_of_two_Gaussians_transitional,
            self.nonsymmetric_mixture_of_two_Gaussians_unimodal,
            self.symmetric_mixture_of_four_Gaussians_multimodal,
            self.symmetric_mixture_of_four_Gaussians_transitional,
            self.symmetric_mixture_of_four_Gaussians_unimodal,
            self.nonsymmetric_mixture_of_four_Gaussians_multimodal,
            self.nonsymmetric_mixture_of_four_Gaussians_transitional,
            self.nonsymmetric_mixture_of_four_Gaussians_unimodal,
        ]

    def generate_noise(self, n_samples, var=None):
        """Generates a random non-Gaussian noise vector."""
        if var is None:
            var = np.random.uniform(1, 3) # Random variance between 1 and 3

        dist_func = np.random.choice(self.distributions)
        noise = dist_func(n_samples)
        
        return (noise / np.std(noise)) * np.sqrt(var)

    def student_3_dof(self, n): return t(3).rvs(n)
    def double_exponential(self, n): return laplace().rvs(n)
    def uniform(self, n): return uniform(-np.sqrt(3), 2 * np.sqrt(3)).rvs(n)
    def student_5_dof(self, n): return t(5).rvs(n)
    def exponential(self, n): return expon().rvs(n) - 1
    def mixture_of_double_exponentials(self, n):
        lap1 = laplace(loc=-3).rvs(n)
        lap2 = laplace(loc=3).rvs(n)
        return lap1 * np.random.binomial(1, 0.5, n) + lap2 * (1 - np.random.binomial(1, 0.5, n))
    def symmetric_mixture_of_two_Gaussians_multimodal(self, n):
        norm1 = norm(-3, 1).rvs(n)
        norm2 = norm(3, 1).rvs(n)
        return norm1 * np.random.binomial(1, 0.5, n) + norm2 * (1 - np.random.binomial(1, 0.5, n))
    def symmetric_mixture_of_two_Gaussians_transitional(self, n):
        norm1 = norm(-1.5, 1).rvs(n)
        norm2 = norm(1.5, 1).rvs(n)
        return norm1 * np.random.binomial(1, 0.5, n) + norm2 * (1 - np.random.binomial(1, 0.5, n))
    def symmetric_mixture_of_two_Gaussians_unimodal(self, n):
        norm1 = norm(-1, 1).rvs(n)
        norm2 = norm(1, 1).rvs(n)
        return norm1 * np.random.binomial(1, 0.5, n) + norm2 * (1 - np.random.binomial(1, 0.5, n))
    def nonsymmetric_mixture_of_two_Gaussians_multimodal(self, n):
        norm1 = norm(-3, 1).rvs(n)
        norm2 = norm(3, 1).rvs(n)
        return norm1 * np.random.binomial(1, 0.25, n) + norm2 * (1 - np.random.binomial(1, 0.25, n))
    def nonsymmetric_mixture_of_two_Gaussians_transitional(self, n):
        norm1 = norm(-1.5, 1).rvs(n)
        norm2 = norm(1.5, 1).rvs(n)
        return norm1 * np.random.binomial(1, 0.25, n) + norm2 * (1 - np.random.binomial(1, 0.25, n))
    def nonsymmetric_mixture_of_two_Gaussians_unimodal(self, n):
        norm1 = norm(-1, 1).rvs(n)
        norm2 = norm(1, 1).rvs(n)
        return norm1 * np.random.binomial(1, 0.25, n) + norm2 * (1 - np.random.binomial(1, 0.25, n))
    def symmetric_mixture_of_four_Gaussians_multimodal(self, n):
        norms = np.vstack([norm(mu, 1).rvs(n) for mu in [-6, -2, 2, 6]])
        mask = np.eye(4)[np.random.choice(4, n, p=[0.15, 0.35, 0.35, 0.15])]
        return (norms.T * mask).sum(axis=1)
    def symmetric_mixture_of_four_Gaussians_transitional(self, n):
        norms = np.vstack([norm(mu, 1).rvs(n) for mu in [-4.5, -1, 1, 4.5]])
        mask = np.eye(4)[np.random.choice(4, n, p=[0.15, 0.35, 0.35, 0.15])]
        return (norms.T * mask).sum(axis=1)
    def symmetric_mixture_of_four_Gaussians_unimodal(self, n):
        norms = np.vstack([norm(mu, 1).rvs(n) for mu in [-3.8, -1, 1, 3.8]])
        mask = np.eye(4)[np.random.choice(4, n, p=[0.15, 0.35, 0.35, 0.15])]
        return (norms.T * mask).sum(axis=1)
    def nonsymmetric_mixture_of_four_Gaussians_multimodal(self, n):
        norms = np.vstack([norm(mu, 1).rvs(n) for mu in [-6, -2, 1.25, 6]])
        mask = np.eye(4)[np.random.choice(4, n, p=[0.20, 0.40, 0.20, 0.20])]
        return (norms.T * mask).sum(axis=1)
    def nonsymmetric_mixture_of_four_Gaussians_transitional(self, n):
        norms = np.vstack([norm(mu, 1).rvs(n) for mu in [-4.5, -1.2, 1, 4.5]])
        mask = np.eye(4)[np.random.choice(4, n, p=[0.2, 0.3, 0.4, 0.1])]
        return (norms.T * mask).sum(axis=1)
    def nonsymmetric_mixture_of_four_Gaussians_unimodal(self, n):
        norms = np.vstack([norm(mu, 1).rvs(n) for mu in [-3, -1, 1, 3]])
        mask = np.eye(4)[np.random.choice(4, n, p=[0.15, 0.15, 0.35, 0.35])]
        return (norms.T * mask).sum(axis=1)

# --- CHANGE: Updated function signature to accept noise_type ---
def generate_synthetic_dataset(n_nodes, n_samples, graph_density, n_confounders, confounder_strength=1.0, is_linear=False, noise_type='nongaussian'):
    """
    Generates synthetic data for causal discovery.

    Args:
        n_nodes (int): Number of variables.
        n_samples (int): Number of data points.
        graph_density (float): Expected number of parents per node.
        n_confounders (int): Number of latent confounders.
        confounder_strength (float): Multiplier for the confounding effect.
        is_linear (bool): If True, generates linear data. If False, generates nonlinear data.
        noise_type (str): Type of noise to generate. Either 'nongaussian' or 'gaussian'.
    """
    n_edges = int(graph_density * n_nodes)
    B = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges, weight_range=(0.5, 1.5))
    G = nx.DiGraph(B)
    causal_order = list(nx.topological_sort(G))
    
    # --- CHANGE: Logic to handle different noise types ---
    if noise_type == 'nongaussian':
        noise_generator = NonGaussianDistributions()
        e = np.array([noise_generator.generate_noise(n_samples) for _ in range(n_nodes)]).T
        f = np.array([noise_generator.generate_noise(n_samples) for _ in range(n_confounders)]).T
    elif noise_type == 'gaussian':
        e = np.array([norm(loc=0, scale=np.random.uniform(1, 3)).rvs(n_samples) for _ in range(n_nodes)]).T
        f = np.array([norm(loc=0, scale=np.random.uniform(1, 3)).rvs(n_samples) for _ in range(n_confounders)]).T
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Must be 'nongaussian' or 'gaussian'.")

    confounding_matrix = np.zeros((n_nodes, n_confounders))
    confounded_variables = {}
    
    if n_confounders > 0:
        nodes = list(range(n_nodes))
        for i in range(n_confounders):
            num_affected = np.random.randint(2, n_nodes + 1)
            affected_nodes = np.random.choice(nodes, size=num_affected, replace=False)
            confounded_variables[f'f{i}'] = list(affected_nodes)
            strengths = np.random.uniform(0.5, 1.5, size=num_affected) * np.random.choice([-1, 1], size=num_affected)
            confounding_matrix[affected_nodes, i] = strengths

    total_confounding_effect = (f @ confounding_matrix.T) * confounder_strength
    X = np.zeros((n_samples, n_nodes))
    
    # --- CHANGE: Define a Gaussian Process for nonlinear data generation ---
    if not is_linear:
        # A standard RBF kernel for smooth, complex functions
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel)

    for i in causal_order:
        parents = np.where(B[:, i] != 0)[0]
        parent_effect = 0
        if len(parents) > 0:
            if is_linear:
                parent_effect = X[:, parents] @ B[parents, i]
            else:
                # --- CHANGE: Use Gaussian Process to generate parent effect ---
                # We use the GP to sample a random nonlinear function of the parents.
                # The shape of parent data for the GP needs to be (n_samples, n_parents)
                parent_data = X[:, parents]
                if parent_data.ndim == 1:
                    parent_data = parent_data.reshape(-1, 1)
                
                # We add a small amount of noise to the GP to ensure stability
                parent_effect = gp.sample_y(parent_data, 1, random_state=None).flatten()

        X[:, i] = parent_effect + total_confounding_effect[:, i] + e[:, i]

    # --- CHANGE: StandardScaler has been removed ---
    # The function now returns the raw generated data.
    X_df = pd.DataFrame(X, columns=[f'x{i}' for i in range(n_nodes)])

    return {'X': X_df, 'B': B, 'causal_order': causal_order, 'confounding_matrix': confounding_matrix, 'confounded_variables': confounded_variables}

def save_dataset(dataset, path):
    os.makedirs(path, exist_ok=True)
    dataset['X'].to_csv(os.path.join(path, 'data.csv'), index=False)
    np.save(os.path.join(path, 'adj_matrix.npy'), dataset['B'])
    np.save(os.path.join(path, 'causal_order.npy'), np.array(dataset['causal_order']))
    np.save(os.path.join(path, 'confounding_matrix.npy'), dataset['confounding_matrix'])
    with open(os.path.join(path, 'confounding_info.txt'), 'w') as f:
        f.write("Confounded Variables:\n")
        if dataset['confounded_variables']:
            for key, val in dataset['confounded_variables'].items():
                f.write(f"- {key}: affects nodes {val}\n")
        else:
            f.write("None\n")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic causal datasets for multiple trials.")
    parser.add_argument('--nodes', type=int, default=10, help='Number of variables.')
    parser.add_argument('--samples', type=int, default=2000, help='Number of samples.')
    parser.add_argument('--density', type=float, default=2, help='Expected number of parents per node.')
    parser.add_argument('--confounders', type=int, default=2, help='Number of latent confounders.')
    parser.add_argument('--confounder_strength', type=float, default=1.5, help='Multiplier for the confounding effect.')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials to generate.')
    parser.add_argument('--out', type=str, default='experiments', help='Base output directory.')
    # --- CHANGE: Modified arguments to select simulation type ---
    parser.add_argument('--scenario', type=str, default='linear-nongaussian', 
                        choices=['linear-nongaussian', 'nonlinear-gaussian'],
                        help='The simulation scenario to generate data for.')
    args = parser.parse_args()

    # Determine settings from scenario
    if args.scenario == 'linear-nongaussian':
        is_linear = True
        noise_type = 'nongaussian'
    elif args.scenario == 'nonlinear-gaussian':
        is_linear = False
        noise_type = 'gaussian'

    exp_name = f"{args.scenario}_n{args.nodes}_s{args.samples}_d{args.density}_c{args.confounders}_cs{args.confounder_strength}_t{args.trials}"
    base_path = os.path.join(args.out, exp_name)
    
    print(f"Generating {args.trials} synthetic dataset trials for scenario '{args.scenario}' in '{base_path}'...")
    
    for i in range(args.trials):
        print(f"- Generating trial {i+1}/{args.trials}...")
        trial_path = os.path.join(base_path, f"trial_{i:03d}")
        dataset = generate_synthetic_dataset(
            args.nodes, args.samples, args.density, args.confounders, 
            args.confounder_strength, is_linear=is_linear, noise_type=noise_type
        )
        save_dataset(dataset, trial_path)
        
    print("Done.")

if __name__ == "__main__":
    main()
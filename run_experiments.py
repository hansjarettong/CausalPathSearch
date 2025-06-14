import os
import argparse
import pandas as pd
import numpy as np
from itertools import combinations
import warnings
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# Suppress scikit-learn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# --- Import Causal Discovery Models ---
from lingam.direct_lingam import DirectLiNGAM
from lingam import RESIT
from lingam_mmi import HSIC_LiNGAM_MMI
from resit_mmi import HSIC_RESIT_MMI
from bidirectional_mmi import BidirectionalMMI 
from hsic_direct_lingam import HSICDirectLiNGAM 

# Regressors for the models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# --- Import Metrics from user-provided file ---
# This uses the cdt library's SID, which is the standard.
# It requires 'pip install cdt' and a working R environment.
try:
    from cdt.metrics import SID
except ImportError:
    print("Warning: Could not import SID from 'metrics.py'. The 'cdt' library might be missing.")
    print("Please run 'pip install cdt' and ensure you have a working R environment.")
    SID = None


def calculate_ordering_error(true_order: list, pred_order: list) -> float:
    """Calculates the fraction of incorrect pairwise orderings."""
    if len(true_order) != len(pred_order):
        print(f"  ! Warning: Predicted order length ({len(pred_order)}) does not match true order length ({len(true_order)}). Assigning max error.")
        return 1.0
    n_nodes = len(true_order)
    if n_nodes < 2: return 0.0
    true_pos = {node: i for i, node in enumerate(true_order)}
    pred_pos = {node: i for i, node in enumerate(pred_order)}
    disagreements = 0
    node_pairs = list(combinations(range(n_nodes), 2))
    for i, j in node_pairs:
        true_i_before_j = true_pos[i] < true_pos[j]
        pred_i_before_j = pred_pos[i] < pred_pos[j]
        if true_i_before_j != pred_i_before_j:
            disagreements += 1
    return disagreements / len(node_pairs)

def calculate_shd(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    """Calculates the Structural Hamming Distance (SHD)."""
    diff = np.abs(true_adj - pred_adj)
    # The SHD for DAGs is the number of edge additions, deletions, or reversals.
    # A reversal is double-counted by a simple diff (one addition, one deletion).
    reversals = np.sum((true_adj.T == 1) & (diff == 1)) / 2
    return int(np.sum(diff) - reversals)

def run_single_trial(trial_path: str, models: dict, use_linear_resit: bool = False) -> dict:
    """Runs all specified models on a single trial's data and returns the results."""
    print(f"Processing {os.path.basename(trial_path)}...")
    data_path = os.path.join(trial_path, 'data.csv')
    order_truth_path = os.path.join(trial_path, 'causal_order.npy')
    adj_truth_path = os.path.join(trial_path, 'adj_matrix.npy')

    if not all(os.path.exists(p) for p in [data_path, order_truth_path, adj_truth_path]):
        print(f"  - Skipping trial, missing files in {trial_path}")
        return None

    data = pd.read_csv(data_path)
    true_order = np.load(order_truth_path).tolist()
    true_adj = (np.load(adj_truth_path) != 0).astype(int)
    
    trial_results = {'trial': os.path.basename(trial_path)}
    resit_regressor = LinearRegression() if use_linear_resit else GradientBoostingRegressor()

    for name, model in models.items():
        print(f"  - Running {name} on {os.path.basename(trial_path)}...")
        start_time = time.time()
        try:
            if name == 'DirectLiNGAM': current_model = DirectLiNGAM()
            elif name == 'HSICDirectLiNGAM': current_model = HSICDirectLiNGAM()
            elif name == 'RESIT': current_model = RESIT(regressor=resit_regressor)
            elif name == 'HSIC_LiNGAM_MMI': current_model = HSIC_LiNGAM_MMI()
            elif name == 'HSIC_RESIT_MMI': current_model = HSIC_RESIT_MMI(regressor=resit_regressor)
            elif name == 'BidirectionalMMI': current_model = BidirectionalMMI(regressor_bwd=resit_regressor)
            
            current_model.fit(data)
            pred_order = current_model.causal_order_
            pred_adj = (current_model.adjacency_matrix_ != 0).astype(int)
            
            error = calculate_ordering_error(true_order, pred_order)
            shd = calculate_shd(true_adj, pred_adj)
            sid = SID(true_adj, pred_adj) if SID is not None else -1.0 # Use -1 to indicate unavailable

        except Exception as e:
            print(f"    ! Error running {name} on {os.path.basename(trial_path)}: {e}")
            error, shd, sid = 1.0, np.sum(true_adj), -1.0

        end_time = time.time()
        
        trial_results[f'{name}_error'] = error
        trial_results[f'{name}_shd'] = shd
        trial_results[f'{name}_sid'] = sid
        trial_results[f'{name}_time_s'] = end_time - start_time

    return trial_results

def main():
    parser = argparse.ArgumentParser(description="Run causal discovery experiments on synthetic datasets.")
    parser.add_argument('--exp_dir', type=str, required=True, help='The base directory of the experiment.')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 1), help='Number of worker processes.')
    parser.add_argument('--resit_linear', action='store_true', help='Use Linear Regression for RESIT models.')
    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        print(f"Error: Experiment directory not found at '{args.exp_dir}'")
        return

    models_to_run = {
        'DirectLiNGAM': None, 'HSICDirectLiNGAM': None, 'RESIT': None,
        'HSIC_LiNGAM_MMI': None, 'HSIC_RESIT_MMI': None, 'BidirectionalMMI': None
    }
    trial_dirs = sorted([d for d in os.listdir(args.exp_dir) if d.startswith('trial_')])
    trial_paths = [os.path.join(args.exp_dir, d) for d in trial_dirs if os.path.isdir(os.path.join(args.exp_dir, d))]

    if not trial_paths:
        print(f"Error: No 'trial_*' subdirectories found in '{args.exp_dir}'.")
        return

    print(f"\nFound {len(trial_paths)} trials. Starting parallel experiments with {args.workers} workers...")
    if args.resit_linear: print(">>> Using LINEAR regressor for all RESIT-based models. <<<")

    worker_func = partial(run_single_trial, models=models_to_run, use_linear_resit=args.resit_linear)
    with Pool(processes=args.workers) as pool:
        all_results = pool.map(worker_func, trial_paths)

    all_results = [res for res in all_results if res is not None]
    if not all_results:
        print("\nNo trials were successfully processed.")
        return

    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(args.exp_dir, 'results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

    summary_data = []
    for model_name in models_to_run.keys():
        summary_data.append({
            'Algorithm': model_name,
            'Ordering Error': f"{results_df[f'{model_name}_error'].mean():.2f} ± {results_df[f'{model_name}_error'].std():.2f}",
            'SHD': f"{results_df[f'{model_name}_shd'].mean():.2f} ± {results_df[f'{model_name}_shd'].std():.2f}",
            'SID': f"{results_df[f'{model_name}_sid'].mean():.2f} ± {results_df[f'{model_name}_sid'].std():.2f}" if SID is not None else "N/A",
            'Time (s)': f"{results_df[f'{model_name}_time_s'].mean():.2f} ± {results_df[f'{model_name}_time_s'].std():.2f}",
        })
        
    summary_df = pd.DataFrame(summary_data).set_index('Algorithm')
    
    print("\n--- Experiment Summary ---")
    print(f"Directory: {args.exp_dir}")
    print(f"Total Trials Processed: {len(results_df)}")
    print("\nAlgorithm Performance (Mean ± Std):")
    print(summary_df.to_string())
    print("\nNote: Lower is better for all metrics (Ordering Error, SHD, SID).")

if __name__ == '__main__':
    main()

import os
import subprocess
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
from functools import partial
from multiprocessing import Pool
import time

# --- Import Algorithm Classes ---
# Baselines
from lingam import DirectLiNGAM, RESIT
from hsic_direct_lingam import HSICDirectLiNGAM
from sklearn.ensemble import GradientBoostingRegressor

# Beam Search Implementations
from beam_search import HSIC_LiNGAM_Beam, HSIC_RESIT_Beam, BidirectionalBeamSearch

# --- Import Metrics & Data Handler ---
from run_experiments import calculate_ordering_error, calculate_shd
from data_handler import generate_synthetic_dataset
try:
    from cdt.metrics import SID
    cdt_available = True
except ImportError:
    cdt_available = False


# ==============================================================================
# Beam Search Comparison Experiment Script
#
# This script compares baseline algorithms against their beam search counterparts,
# focusing on performance across different sequence lengths (n_nodes).
#
# Usage:
#   nohup python3 run_beam_comparison.py > /dev/null 2>&1 &
# ==============================================================================

# --- Configuration ---
NUM_TRIALS = 30
NUM_WORKERS = 20
BASE_EXPERIMENTS_DIR = "beam_comparison_experiments"

# --- Experiment Parameter Space ---
# Focused on testing the "longer sequences" hypothesis
PARAM_GRID = {
    "n_nodes": [5, 10, 15, 30],
    "n_samples": [500],
    "beam_width": [5,10], # Fixed beam width for this experiment
    "scenario": ["nonlinear-gaussian"], # Most challenging scenario
    "density": [2.0],
    "n_confounders": [2],
    "conf_strength": [1.5],
}

# --- Model Definitions ---
# Defines all models to be run in the experiment
MODELS_TO_RUN = {
    "DirectLiNGAM": {"class": DirectLiNGAM, "params": {}},
    "HSICDirectLiNGAM": {"class": HSICDirectLiNGAM, "params": {}},
    "RESIT": {"class": RESIT, "params": {"regressor": GradientBoostingRegressor(n_estimators=20)}},
    "HSIC_LiNGAM_Beam": {"class": HSIC_LiNGAM_Beam, "params": {"beam_width": 5}},
    "HSIC_RESIT_Beam": {"class": HSIC_RESIT_Beam, "params": {"regressor": GradientBoostingRegressor(n_estimators=20), "beam_width": 5}},
    "BidirectionalBeamSearch": {"class": BidirectionalBeamSearch, "params": {"regressor_bwd": GradientBoostingRegressor(n_estimators=20), "beam_width": 5}},
}


# --- Setup Master Run Directory and Logging ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_RUN_DIR = os.path.join(BASE_EXPERIMENTS_DIR, f"run_{TIMESTAMP}")
os.makedirs(MASTER_RUN_DIR, exist_ok=True)

LOG_FILE = os.path.join(MASTER_RUN_DIR, f"beam_comparison_{TIMESTAMP}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

# --- Master Summary File ---
MASTER_SUMMARY_FILE = os.path.join(MASTER_RUN_DIR, "master_summary.csv")
# Initialize master summary file
pd.DataFrame(columns=[
    'n_nodes', 'n_samples', 'beam_width', 'algorithm', 'trial',
    'ordering_error', 'shd', 'sid', 'time_s'
]).to_csv(MASTER_SUMMARY_FILE, index=False)
logging.info(f"Master run directory: {MASTER_RUN_DIR}")
logging.info(f"Master summary file: {MASTER_SUMMARY_FILE}")


def run_single_trial(trial_num, config, model_name):
    """
    Generates data for one trial, runs a single model, and returns results.
    This function is designed to be called by the multiprocessing pool.
    """
    try:
        # --- 1. Generate Data ---
        # Ensure each trial has unique data by seeding with the trial number
        np.random.seed(int(time.time()) + trial_num * 100)
        
        data_params = {
            "n_nodes": config["n_nodes"], "n_samples": config["n_samples"],
            "graph_density": config["density"], "n_confounders": config["n_confounders"],
            "confounder_strength": config["conf_strength"], "is_linear": False,
            "noise_type": 'gaussian'
        }
        dataset = generate_synthetic_dataset(**data_params)
        X_df = dataset['X']
        true_order = dataset['causal_order']
        true_adj = (dataset['B'] != 0).astype(int)

        # --- 2. Run Model ---
        model_info = MODELS_TO_RUN[model_name]
        model = model_info["class"](**model_info["params"])
        
        start_time = time.time()
        model.fit(X_df)
        end_time = time.time()
        
        pred_order = model.causal_order_
        pred_adj = (model.adjacency_matrix_ != 0).astype(int)

        # --- 3. Calculate Metrics ---
        error = calculate_ordering_error(true_order, pred_order)
        shd = calculate_shd(true_adj, pred_adj)
        sid = SID(true_adj, pred_adj) if cdt_available else -1.0

        result = {
            'n_nodes': config['n_nodes'], 'n_samples': config['n_samples'],
            'beam_width': config.get('beam_width', -1), 'algorithm': model_name,
            'trial': trial_num, 'ordering_error': error, 'shd': shd,
            'sid': sid, 'time_s': end_time - start_time
        }
        return result
    except Exception as e:
        logging.error(f"Trial {trial_num} for {model_name} failed with error: {e}", exc_info=True)
        return None

def run_configuration(config):
    """Manages all trials for a single experimental configuration."""
    exp_name_parts = [
        f"n{config['n_nodes']}", f"s{config['n_samples']}", f"k{config['beam_width']}"
    ]
    exp_name = "_".join(exp_name_parts)
    logging.info("=" * 70)
    logging.info(f"STARTING CONFIGURATION: {exp_name}")
    logging.info(f"Parameters: {config}")
    logging.info("-" * 70)

    all_results = []
    
    # Iterate through each model
    for model_name in MODELS_TO_RUN.keys():
        logging.info(f"--- Running model: {model_name} for config: {exp_name} ---")
        
        # Use a multiprocessing pool to run trials in parallel
        with Pool(processes=NUM_WORKERS) as pool:
            # Create a partial function to pass fixed arguments to the worker
            worker_func = partial(run_single_trial, config=config, model_name=model_name)
            # Map the function over the trial numbers
            results = pool.map(worker_func, range(NUM_TRIALS))
            
            # Filter out any failed trials
            successful_results = [res for res in results if res is not None]
            all_results.extend(successful_results)
            logging.info(f"Completed {len(successful_results)}/{NUM_TRIALS} trials for {model_name}.")

    # --- Aggregate and Save Results for this Configuration ---
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(MASTER_SUMMARY_FILE, mode='a', header=False, index=False)
        logging.info(f"Appended {len(df_results)} results to master summary file.")
    
    logging.info("-" * 70)
    logging.info(f"FINISHED CONFIGURATION: {exp_name}")
    logging.info("=" * 70 + "\n")


def main():
    """Main execution logic to iterate through all experiment configurations."""
    logging.info("############################################################")
    logging.info("###   STARTING BEAM SEARCH VS BASELINES COMPARISON       ###")
    logging.info("############################################################\n")

    # Generate the list of all experimental configurations
    keys, values = zip(*PARAM_GRID.items())
    experiment_configs = [dict(zip(keys, v)) for v in product(*values)]

    for config in experiment_configs:
        run_configuration(config)

    logging.info("############################################################")
    logging.info("###   ALL BEAM SEARCH EXPERIMENTS HAVE COMPLETED         ###")
    logging.info("############################################################")
    logging.info(f"Master summary file is located at: {MASTER_SUMMARY_FILE}")
    logging.info(f"Detailed logs are in: {LOG_FILE}")

if __name__ == "__main__":
    main()


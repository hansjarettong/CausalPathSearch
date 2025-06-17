import os
import subprocess
import logging
import pandas as pd
from datetime import datetime
from itertools import product

# ==============================================================================
# Master Experiment Automation Script (Python Version)
#
# This script systematically generates data and runs experiments for all
# specified configurations for the CausalPathSearch project.
#
# Usage:
#   nohup python3 run_pipeline.py > /dev/null 2>&1 &
#   (All output is handled by the logger, so stdout/stderr can be discarded)
# ==============================================================================

# --- Configuration ---
NUM_TRIALS = 30
NUM_WORKERS = 20
BASE_EXPERIMENTS_DIR = "paper_experiments"

# --- Experiment Parameter Space ---
# Each dictionary represents a set of experiments to run.
# The script will iterate through the Cartesian product of all list values.
EXPERIMENT_SETS = {
    "Core": {
        "scenario": ["linear-nongaussian", "nonlinear-gaussian"],
        "n_nodes": [10],
        "n_samples": [1000],
        "density": [2.0],
        "n_confounders": [2],
        "conf_strength": [1.5],
    },
    "Challenging_Nodes": {
        "scenario": ["nonlinear-gaussian"],
        "n_nodes": [5, 15],
        "n_samples": [1000],
        "density": [2.0],
        "n_confounders": [2],
        "conf_strength": [1.5],
    },
    "Challenging_Samples": {
        "scenario": ["nonlinear-gaussian"],
        "n_nodes": [10],
        "n_samples": [100, 500],
        "density": [2.0],
        "n_confounders": [2],
        "conf_strength": [1.5],
    },
    "Challenging_Density": {
        "scenario": ["nonlinear-gaussian"],
        "n_nodes": [10],
        "n_samples": [1000],
        "density": [3.0],
        "n_confounders": [2],
        "conf_strength": [1.5],
    },
    "Challenging_Confounders": {
        "scenario": ["nonlinear-gaussian"],
        "n_nodes": [10],
        "n_samples": [1000],
        "density": [2.0],
        "n_confounders": [0, 4],
        "conf_strength": [1.5],
    },
    "Challenging_Confounder_Strength": {
        "scenario": ["nonlinear-gaussian"],
        "n_nodes": [10],
        "n_samples": [1000],
        "density": [2.0],
        "n_confounders": [2],
        "conf_strength": [2.5],
    },
}

# --- Setup Master Run Directory and Logging ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_RUN_DIR = os.path.join(BASE_EXPERIMENTS_DIR, f"run_{TIMESTAMP}")
os.makedirs(MASTER_RUN_DIR, exist_ok=True)

LOG_FILE = os.path.join(MASTER_RUN_DIR, f"run_pipeline_{TIMESTAMP}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# --- Master Summary File ---
MASTER_SUMMARY_FILE = os.path.join(MASTER_RUN_DIR, "master_summary_long.csv")
# Initialize master summary file with a header
pd.DataFrame(columns=[
    'trial_id', 'scenario', 'n_nodes', 'n_samples', 'density', 'n_confounders',
    'conf_strength', 'resit_linear', 'algorithm', 'ordering_error', 'shd', 'sid', 'time_s'
]).to_csv(MASTER_SUMMARY_FILE, index=False)
logging.info(f"Master run directory created at: {MASTER_RUN_DIR}")
logging.info(f"Master summary file initialized at: {MASTER_SUMMARY_FILE}")

def run_command(command):
    """Executes a command and logs its output."""
    logging.info(f"Executing command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Command successful. Output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}.")
        logging.error(f"Stderr:\n{e.stderr}")
        logging.error(f"Stdout:\n{e.stdout}")
        return False

def aggregate_results(results_csv_path, params):
    """Reads a results.csv, transforms it to long format, and appends to master summary."""
    try:
        df = pd.read_csv(results_csv_path)
        
        # Melt the DataFrame from wide to long format
        id_vars = ['trial']
        value_vars = [col for col in df.columns if col != 'trial']
        df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='metric_info', value_name='value')
        
        # Extract algorithm and metric type from the 'metric_info' column
        df_long['algorithm'] = df_long['metric_info'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        df_long['metric'] = df_long['metric_info'].apply(lambda x: x.split('_')[-1])
        
        # Pivot the table to get metrics as columns
        df_pivot = df_long.pivot_table(index=['trial', 'algorithm'], columns='metric', values='value').reset_index()
        df_pivot.rename(columns={'error': 'ordering_error', 's': 'time_s'}, inplace=True)
        
        # Add parameter columns
        for key, val in params.items():
            df_pivot[key] = val
        
        # Reorder columns to match the master summary
        final_cols = pd.read_csv(MASTER_SUMMARY_FILE).columns
        df_pivot = df_pivot.rename(columns={'trial':'trial_id'})
        df_pivot = df_pivot[final_cols]

        # Append to master CSV
        df_pivot.to_csv(MASTER_SUMMARY_FILE, mode='a', header=False, index=False)
        logging.info(f"Successfully aggregated results from {results_csv_path}")

    except Exception as e:
        logging.error(f"Failed to aggregate results from {results_csv_path}. Error: {e}")

def run_configuration(params):
    """Main function to run one full experiment configuration."""
    
    # --- Build Experiment Name ---
    resit_mode = "linear-resit" if params['resit_linear'] else "nonlinear-resit"
    exp_name_parts = [
        params['scenario'],
        f"n{params['n_nodes']}", f"s{params['n_samples']}", f"d{params['density']}",
        f"c{params['n_confounders']}", f"cs{params['conf_strength']}",
        f"t{NUM_TRIALS}", resit_mode
    ]
    exp_name = "_".join(map(str, exp_name_parts))
    exp_dir = os.path.join(MASTER_RUN_DIR, exp_name)

    logging.info("=" * 70)
    logging.info(f"STARTING EXPERIMENT: {exp_name}")
    logging.info(f"Parameters: {params}")
    logging.info("-" * 70)

    # --- 1. Generate Data ---
    data_gen_dir_name = "_".join(map(str, exp_name_parts[:-1])) # Name without resit mode
    data_gen_base_dir = os.path.join(exp_dir, data_gen_dir_name)

    cmd_data = [
        "python3", "data_handler.py",
        "--scenario", params['scenario'],
        "--nodes", str(params['n_nodes']),
        "--samples", str(params['n_samples']),
        "--density", str(params['density']),
        "--confounders", str(params['n_confounders']),
        "--confounder_strength", str(params['conf_strength']),
        "--trials", str(NUM_TRIALS),
        "--out", exp_dir
    ]
    if not run_command(cmd_data):
        logging.error(f"Data generation failed for {exp_name}. Skipping run.")
        return

    # --- 2. Run Experiments ---
    cmd_run = [
        "python3", "run_experiments.py",
        "--exp_dir", data_gen_base_dir,
        "--workers", str(NUM_WORKERS)
    ]
    if params['resit_linear']:
        cmd_run.append("--resit_linear")

    if not run_command(cmd_run):
        logging.error(f"Experiment run failed for {exp_name}.")
        return

    # --- 3. Aggregate Results ---
    results_csv_path = os.path.join(data_gen_base_dir, "results.csv")
    if os.path.exists(results_csv_path):
        aggregate_results(results_csv_path, params)
    else:
        logging.warning(f"Results file not found for {exp_name} at {results_csv_path}")

    logging.info("-" * 70)
    logging.info(f"FINISHED EXPERIMENT: {exp_name}")
    logging.info("=" * 70 + "\n")

def main():
    """Main execution logic to iterate through all experiment sets."""
    logging.info("##################################################")
    logging.info("###   STARTING FULL EXPERIMENTAL PIPELINE      ###")
    logging.info("##################################################\n")

    for stage_name, param_set in EXPERIMENT_SETS.items():
        logging.info(f"\n================> STAGE: {stage_name} <================")
        
        # Get all combinations of parameters for the current set
        keys, values = zip(*param_set.items())
        for v in product(*values):
            params = dict(zip(keys, v))
            
            # Run for both resit_linear True and False
            for resit_linear_bool in [True, False]:
                run_params = params.copy()
                run_params['resit_linear'] = resit_linear_bool
                run_configuration(run_params)

    logging.info("##################################################")
    logging.info("###   ALL EXPERIMENTS HAVE COMPLETED           ###")
    logging.info("##################################################")
    logging.info(f"Master summary file is located at: {MASTER_SUMMARY_FILE}")
    logging.info(f"Detailed logs are in: {LOG_FILE}")

if __name__ == "__main__":
    main()

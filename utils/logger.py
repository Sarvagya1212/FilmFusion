import os
import logging
from datetime import datetime

# --- Configuration ---
# Define the log directory and file path once as constants
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "evaluation_log.txt")

def setup_logger():
    """
    Configures a global logger to append to the evaluation log file.
    This setup function ensures all subsequent log calls write to the same file.
    """
    # Ensure the 'logs' directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Use Python's built-in logging module
    # 'a' mode ensures that we always append to the file
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # We'll format the message ourselves before logging
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
            logging.StreamHandler()  # This also prints the log to your console
        ]
    )

# This function should be called by your evaluation script
def log_evaluation_results(metrics, model_name, top_k, features_used, extra_params):
    """
    Formats and logs evaluation results using the configured logger.
    """
    logger = logging.getLogger(__name__)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create the log message as a list of strings for clarity
    log_entry = [
        f"\n----- Evaluation @ {timestamp} -----",
        f"Model Name: {model_name}",
        f"Top@K: {top_k}",
        f"Features Used: {', '.join(features_used) if isinstance(features_used, list) else features_used}",
    ]

    # Add all extra parameters (like strategy, alpha, etc.)
    if extra_params:
        log_entry.extend(f"{key}: {value}" for key, value in extra_params.items())

    # Add all performance metrics
    if metrics:
        log_entry.extend(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}" for key, value in metrics.items())

    log_entry.append("-------------------------------------")

    # Join the list into a single string and log it
    logger.info("\n".join(log_entry))
    # The print statement is no longer needed as the StreamHandler will output to console
    # print(f"Evaluation results logged to: {LOG_FILE}")

# Set up the logger as soon as this module is imported
setup_logger()
import os
import subprocess
import authpipe.configuration.configuration as config
from authpipe.data_processing_utilities.data_processing_utilities import load_features
from authpipe.training_utilities.training_utilities import build_authenticator

def run_train(lbound, rbound, window, model, context_path, plot_results, verbose, debug):
  
    n_intervals = None
    if not (rbound is None) and not (lbound is None):
        n_intervals = int((rbound - lbound) / window) + 1
    
    config.settings.context_path = context_path
    config.settings.rbound = rbound
    config.settings.lbound = lbound
    config.settings.window = window
    config.settings.n_intervals = n_intervals
    config.settings.verbose = verbose
    
    load_features()
    
    models_path = os.path.join(context_path, 'models')
    
    if os.path.exists(models_path):
        subprocess.run(['rm', '-rf', models_path])
    subprocess.run(['mkdir', models_path])
        
    build_authenticator(model, plot_results)

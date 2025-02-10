import os
import subprocess
import authpipe.core.configuration as config
from authpipe.core.data_processing import load_sets, extract_features

def run_extract_features(context_path, verbose, debug, falcon_verbose):
  
  config.settings.context_path = context_path
  config.settings.falcon_verbose = falcon_verbose
  
  tops_path = os.path.join(context_path, 'tops/')
  
  if os.path.exists(tops_path):
      subprocess.run(['rm', '-rf', tops_path])
  subprocess.run(['mkdir', '-p', tops_path])
    
  load_sets()
  extract_features()
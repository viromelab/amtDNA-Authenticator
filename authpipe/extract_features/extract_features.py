import os
import subprocess
import authpipe.core.configuration as config
from authpipe.core.data_processing import load_sets, extract_features

def run_extract_features(context_path, verbose, debug, falcon_verbose):
  
  config.settings.context_path = context_path
  config.settings.falcon_verbose = falcon_verbose
    
  load_sets()
  extract_features()
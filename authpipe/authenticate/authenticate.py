import authpipe.core.configuration as config
import authpipe.core.logging as logging
from authpipe.core.data_processing import read_multifasta, get_falcon_scores, integrate_falcon_data, get_falcon_estimations, get_quantitative_data, merge_data, load_features
from authpipe.core.training_core import authentication

def run_authenticate(threshold, model, context_path, auth_path, verbose, debug, single_path, falcon_verbose):
  
  if single_path == None and auth_path == None:
    logging.error(f'No path defined for authentication folder with multi-FASTA nor single FASTA')
    exit()
    
  config.settings.model = model
  config.settings.threshold = threshold
  config.settings.context_path = context_path
  config.settings.auth_path = auth_path
  config.settings.single_path = single_path
  config.settings.falcon_verbose = falcon_verbose
  
  read_multifasta()
  get_falcon_scores()
  integrate_falcon_data()
  get_falcon_estimations()
  get_quantitative_data()
  merge_data()
  load_features()
  authentication()
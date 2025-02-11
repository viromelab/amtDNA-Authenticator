import authpipe.core.configuration as config
from authpipe.core.data_processing import read_multifasta, get_falcon_scores, integrate_falcon_data, get_falcon_estimations, get_quantitative_data, merge_data, load_features
from authpipe.core.training_core import authentication

def run_authenticate(threshold, model, context_path, auth_path, verbose, debug):
  
  config.settings.model = model
  config.settings.threshold = threshold
  config.settings.context_path = context_path
  config.settings.auth_path = auth_path
  
  read_multifasta()
  get_falcon_scores()
  integrate_falcon_data()
  get_falcon_estimations()
  get_quantitative_data()
  merge_data()
  load_features()
  authentication()
import authpipe.core.configuration as config
from authpipe.process_multifasta.process_multifasta import read_multifasta
from authpipe.core.training_core import authentication

def run_authenticate(threshold, model, context_path, samples_path):
  
  config.settings.model = model
  config.settings.threshold = threshold
  config.settings.context_path = context_path
  config.settings.samples_path = samples_path
  
  read_multifasta()
  authentication()
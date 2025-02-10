
import subprocess
import logging
import os
import authpipe.configuration.configuration as config
from authpipe.data_processing_utilities.data_processing_utilities import read_multifasta, divide_set

def run_process_multifasta(multifasta_path, context_path, verbose, debug):
        
    config.settings.context_path = context_path
    
    read_multifasta()
    divide_set()

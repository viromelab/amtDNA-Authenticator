
import subprocess
import logging
import os
import authpipe.core.configuration as config
from authpipe.core.data_processing import read_multifasta, divide_set

def run_process_multifasta(context_path, verbose, debug):
        
    config.settings.context_path = context_path
    
    read_multifasta()
    divide_set()

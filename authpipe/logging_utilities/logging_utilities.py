import logging
import colorlog
import authpipe.configuration.configuration as config

VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")

def verbose(msg, *args, **kwargs):
    if config.settings.verbose:
        logging.log(VERBOSE, msg, *args, **kwargs)
    
def setup_log():
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'VERBOSE':  'blue',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        },
        style='%',
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log_file = "authpipe.log"
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # You could use the same formatter
    file_handler.setFormatter(file_formatter) # But a plain one is often better for files.

    logging.getLogger().addHandler(handler)
    logging.getLogger().addHandler(file_handler)
    logger = logging.getLogger() 
    logger.addHandler(handler)
    
    if config.settings.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(VERBOSE)
    
    logging.verbose = verbose
      
  

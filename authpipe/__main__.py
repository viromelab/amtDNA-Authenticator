import re
import subprocess
import os
import random
import logging
import click
import authpipe.core.configuration as config
from authpipe.core.logging import setup_log
from authpipe.process_multifasta.process_multifasta import run_process_multifasta
from authpipe.extract_features.extract_features import run_extract_features
from authpipe.training.training import run_train
from authpipe.authenticate.authenticate import run_authenticate

# ----------------------------------------------------------------------
# Section: Main
# ----------------------------------------------------------------------

@click.group()
@click.option(
    '-v',
    '--verbose', 
    is_flag=True, 
    help="Verbose mode"
)
@click.option(
    '-d',
    '--debug', 
    is_flag=True, 
    help="Debug mode"
)
@click.pass_context
def main(ctx, verbose, debug, no_args_is_help=True, **kwargs):
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['debug'] = debug
        
    config.settings.execution_path = os.path.dirname(os.path.abspath(__file__))
    config.settings.verbose = verbose
    config.settings.debug = debug

    setup_log()
    
# ----------------------------------------------------------------------
# Section: Process Multifasta
# ----------------------------------------------------------------------

@main.command()
@click.option(
    '-p',
    '--multifasta_path',
    type=click.Path(writable=True, dir_okay=False, file_okay=True),
    show_default=True,
    help='Path to multifasta that will be used in training phase'
)
@click.option(
    '-c',
    '--context_path',
    type=click.Path(writable=True, dir_okay=True, file_okay=False),
    show_default=True,
    help='Path to folder to store/retrieve application context [Will be created if does not exist]', 
    required=True
)
@click.pass_context
def process_multifasta(ctx, no_args_is_help=True, **kwargs):
    """Process, uniformize and divide (Train, Val and Test) multi-FASTA"""
    
    logging.info('Processing Multi-FASTA...')
    
    phase = 'multifasta'
    
    config.settings.phase = phase
    
    run_process_multifasta(**kwargs, **ctx.obj)

# ----------------------------------------------------------------------
# Section: Extract Features 
#   - Multi-FASTA already processed and features ready for extraction
# ----------------------------------------------------------------------

@main.command()
@click.option(
    '-c',
    '--context_path',
    type=click.Path(writable=True, dir_okay=True, file_okay=False),
    show_default=True,
    help='Path to folder to store/retrieve application context [It should exist from previous multi-FASTA processing!]', 
    required=True
)
@click.option(
    '-f',
    '--falcon_verbose',
    is_flag=True,
    help='Show FALCON verbose', 
)
@click.pass_context
def extract_features(ctx, no_args_is_help=True, **kwargs):
    """Extract features from processed sub-groups of multi-FASTA (Train, Val and Test)"""
    
    logging.info('Extracting Features...')
    
    phase = 'extract_features'
    
    config.settings.phase = phase
    
    run_extract_features(**kwargs, **ctx.obj)
    
# ----------------------------------------------------------------------
# Section: Training model 
#   - Train model on features already extracted from multi-FASTAS
# ----------------------------------------------------------------------

@main.command()
@click.option(
    '-l', 
    '--lbound', 
    type=int, 
    help='The min year considered for the threshold sliding')
@click.option(
    '-r',
    '--rbound',
    type=int,
    help='The max year considered for the threshold sliding'
)
@click.option(
    '-w', 
    '--window', 
    type=int, 
    help='The time window to use in training phase [Default: 100] :\n'
         '  - 10: 10 years threshold cuts.\n'
         '  - 100: 100 years threshold cuts.\n'
         '  - 1000: 1000 years threshold cuts.\n')
@click.option(
    '-m',
    '--model',
    type=click.Choice(['XGB', 'KNN', 'NET', 'SVM', 'GNB']),
    help="The model to use in training phase [Default: XGB]:\n"
    "  - XGB: XGBoost.\n"
    "  - KNN: K-Nearest Neighbors.\n"
    "  - NET: Neural Network.\n"
    "  - SVM: Support Vector Machine.\n"
    "  - GNB: Gaussian Naive Bayes.\n",
    default='XGBoost'
)
@click.option(
    '-c',
    '--context_path',
    type=click.Path(writable=True, dir_okay=True, file_okay=False),
    show_default=True,
    help='Path to folder to store/retrieve application context [It should exist from previous multi-FASTA processing!]', 
    required=True
)
@click.option(
    '-p',
    '--plot_results',
    is_flag=True,
    help='Plot results from training phase', 
    required=True
)
@click.pass_context
def train(ctx, no_args_is_help=True, **kwargs):
    """Train model over features extracted from processed sub-groups of multi-FASTA (Train, Val and Test)"""
    
    logging.info('Training Model...')
    
    phase = 'train'
    
    config.settings.phase = phase
    
    run_train(**kwargs, **ctx.obj)
    
# ----------------------------------------------------------------------
# Section: Authenticate 
#   - Authenticate samples
# ----------------------------------------------------------------------

@main.command()
@click.option(
    '-t', 
    '--threshold', 
    type=int, 
    help='Threshold to consider ancient in authentication')
@click.option(
    '-m',
    '--model',
    type=click.Choice(['XGB', 'KNN', 'NET', 'SVM', 'GNB']),
    help="The model to use in training phase [Default: XGB]:\n"
    "  - XGB: XGBoost.\n"
    "  - KNN: K-Nearest Neighbors.\n"
    "  - NET: Neural Network.\n"
    "  - SVM: Support Vector Machine.\n"
    "  - GNB: Gaussian Naive Bayes.\n",
    default='XGBoost'
)
@click.option(
    '-c',
    '--context_path',
    type=click.Path(writable=True, dir_okay=True, file_okay=False),
    show_default=True,
    help='Path to folder to store/retrieve application context [It should exist from previous training!]', 
    required=True
)
@click.option(
    '-p',
    '--auth_path',
    type=click.Path(writable=True, dir_okay=True, file_okay=False),
    show_default=True,
    help='Path to folder with multi-FASTA to authenticate and to save authentication context', 
    required=True
)
@click.pass_context
def authenticate(ctx, no_args_is_help=True, **kwargs):
    """Authenticate samples from FASTA/multi-FASTA file as Modern/Ancient given a threshold age"""
    
    logging.info('Authenticating Samples...')
    
    phase = 'authenticate'
    
    config.settings.phase = phase
    
    run_authenticate(**kwargs, **ctx.obj)
    
if __name__ == '__main__':
    main()

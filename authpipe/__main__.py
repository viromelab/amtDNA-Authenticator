import re
import numpy as np
import joblib
import subprocess
import pandas as pd
import os
import random
import logging
from math import floor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from colorama import Fore, Back, Style
from argparse import ArgumentParser, Namespace
from authpipe.plot_utilities import plot, plot_binary_perf, plot_binary_auroc_auprc, plot_binary_hist_dist
from authpipe.data_processing_utilities import read_multifasta, divide_set, load_sets, get_falcon_scores, integrate_falcon_data, get_falcon_predictions, get_quantitative_data, merge_data, extract_features, load_features

this_file_path = os.path.dirname(os.path.abspath(__file__))

lbound = None
rbound = None

context_path = None

df_train = None
df_val = None
df_test = None

verbose = False

samples = {}
samples_train = {}
samples_test = {}
samples_val = {}

window = None

def train(threshold, model_name):
    global df_train, verbose, context_path

    # Separate features (X) and target (y)
    X_train = df_train[['falcon_estimated_age', 'cg_content', 'relative_size', 'n_content']]
    # X_train = df_train[['falcon_estimated_age']]
    y_train = df_train['age']

    y_train = pd.DataFrame(
        [1 if value >= threshold else 0 for value in y_train])
    y_train_categories = y_train.copy()

    if model_name == 'XGB':
        model = XGBRegressor()
    elif model_name == 'KNN':
        model = KNeighborsRegressor(n_neighbors=40)
    elif model_name == 'SVM':
        model = SVR()
    elif model_name == 'NET':
        model = MLPRegressor(hidden_layer_sizes=(
            100, 100), max_iter=500, random_state=42)
    elif model_name == 'GNB':
        model = GaussianNB()
    else:
        logging.error('Invalid model! Run \'authpipe --help\' to check the models available!')
        exit()

    # Train model
    model.fit(X_train, y_train.values.ravel())

    # Make predictions on testing set
    y_pred = model.predict(X_train)

    # y_pred = pd.DataFrame([1 if pred >= threshold else 0 for pred in y_pred])
    y_pred = pd.DataFrame([1 if pred >= 0.5 else 0 for pred in y_pred])

    # Evaluate model performance
    acc = (y_train == y_pred).mean()
    precision = precision_score(y_train, y_pred, zero_division=0)
    recall = recall_score(y_train, y_pred, zero_division=0)
    f1 = f1_score(y_train, y_pred)
    cm = confusion_matrix(y_train, y_pred, labels=[0, 1])

    # Save the trained model
    model_file_name = os.path.join(context_path, 'models/model_' + model_name + '.joblib')
    joblib.dump(model, model_file_name)

    return acc, precision, recall, f1, y_train, y_pred, cm, y_train_categories


def test(threshold, model_name):
    global df_test, verbose, context_path

    # Separate features (X) and target (y)
    X_test = df_test[['falcon_estimated_age', 'cg_content', 'relative_size', 'n_content']]
    # X_test = df_test[['falcon_estimated_age']]
    y_test = df_test['age']
    # Save the trained model
    model_file_name = os.path.join(
        context_path, 'models/model_' + model_name + '.joblib')
    # Load the saved model
    model = joblib.load(model_file_name)

    y_test_categories = y_test.copy()
    y_test_categories = [int(i/100) for i in y_test_categories]
    y_test = pd.DataFrame([1 if value >= threshold else 0 for value in y_test])

    # Make predictions on testing set
    y_pred = model.predict(X_test)
    # y_pred = pd.DataFrame([1 if pred >= threshold else 0 for pred in y_pred])
    y_pred = pd.DataFrame([1 if pred >= 0.5 else 0 for pred in y_pred])

    # Evaluate model performance
    acc = (y_test == y_pred).mean()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    if y_test.nunique().values[0] > 1:
        auroc = roc_auc_score(y_test, y_pred)
        auprc = average_precision_score(y_test, y_pred)
    else:
        auroc = 0
        auprc = 0

    return acc, precision, recall, f1, y_test, y_pred, cm, y_test_categories, auroc, auprc


def build_authenticator(model_name):
    global window, N_INTERVALS, rbound, lbound
    
    auroc_array = []
    auprc_array = []
    acc_array = []
    precision_array = []
    recall_array = []
    f1_array = []
    always_ancient_random_array = []
    honest_coin_random_array = []
    y_pred_ancient_array = []
    random_array = []
    C00_array = []
    C01_array = []
    C10_array = []
    C11_array = []
    f1_weighted_array = []

    max_lim = int(rbound/window)
    min_lim = int(lbound/window)
    
    N_INTERVALS = max_lim - min_lim + 1

    samples_per_century_array = {i: 0 for i in range(0, N_INTERVALS)}
        
    for threshold in range(min_lim, max_lim):
        if not verbose:
            logging.info(f'\r{threshold}/{N_INTERVALS - 1}')

        train(threshold=window*threshold, model_name=model_name)
        acc, precision, recall, f1, y_true, y_pred, cm, y_true_categories, auroc, auprc = test(
                threshold=window*threshold, model_name=model_name)

        auroc_array.append(auroc)
        auprc_array.append(auprc)
        acc_array.append(acc)
        precision_array.append(precision)
        recall_array.append(recall)
        f1_array.append(f1)
        always_ancient_random_array.append(
            y_true.sum().values[0]/len(y_true))
        honest_coin_random_array.append(y_true.sum().values[0]/len(y_true))
        y_pred_ancient_array.append(y_pred.sum()/len(y_pred))

        num_of_positives = y_true.sum().values[0]
        num_of_samples = len(y_true)

        random_val = max(num_of_positives, num_of_samples -
                            num_of_positives) / num_of_samples

        random_array.append(random_val)
        
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        f1_weighted_array.append(f1_weighted)

        C00_array.append(cm[0][0]/num_of_samples)
        C01_array.append(cm[0][1]/num_of_samples)
        C10_array.append(cm[1][0]/num_of_samples)
        C11_array.append(cm[1][1]/num_of_samples)

        y_true_categories = list(y_true_categories)
        y_true_categories_unique = list(set(y_true_categories))

        samples_per_century = {i: y_true_categories.count(
            i) for i in y_true_categories_unique if i < N_INTERVALS}

        for key in samples_per_century.keys():
            samples_per_century_array[key] += samples_per_century[key]

    total_samples = 0
    for key in samples_per_century_array.keys():
        total_samples += samples_per_century_array[key]

    for key in samples_per_century_array.keys():
        samples_per_century_array[key] = samples_per_century_array[key]/total_samples

    true_predictions = [C00_array[i] + C11_array[i]
                        for i in range(len(C00_array))]
    false_predictions = [C01_array[i] + C10_array[i]
                            for i in range(len(C01_array))]

    plot_binary_perf(acc_array, precision_array, recall_array, f1_weighted_array,
                        honest_coin_random_array, y_pred_ancient_array, false_predictions, random_array)
    plot_binary_auroc_auprc(auroc_array, auprc_array)
    plot_binary_hist_dist(samples_per_century_array)


def main():
    global samples, verbose, context_path, window, rbound, lbound

    logging.basicConfig(
        filename='amtauth.log',  
        level=logging.INFO,      
        format='%(asctime)s - %(levelname)s - %(message)s'  
    )
    
    parser = ArgumentParser()

    parser.add_argument(
        '--phase',
        choices=['multifasta', 'feature_extraction', 'training', 'auth'],
        help="The phase from which to run the program in:\n"
        "  - multifasta: Start with an all-samples multifasta file.\n"
        "  - feature_extraction: Samples are already locally divided into training, validation and test, then extract features.\n"
        "  - training: Sets and features are ready, proceede to train.\n"
        "  - auth: Authenticates a fasta sequence with an already trained model.\n",
        required=True  # Makes the argument mandatory
    )
    
    parser.add_argument(
        '--model',
        choices=['XGB', 'KNN', 'NET', 'SVM', 'GNB'],
        help="The model to use in training phase [Default: XGB]:\n"
        "  - XGB: XGBoost.\n"
        "  - KNN: K-Nearest Neighbors.\n"
        "  - NET: Neural Network.\n"
        "  - SVM: Support Vector Machine.\n"
        "  - GNB: Gaussian Naive Bayes.\n",
        default='XGBoost'
    )
    
    parser.add_argument(
        '--window',
        choices=['10', '100', '1000'],
        help="The time window to use in training phase [Default: 100] :\n"
        "  - 10: 10 years threshold cuts.\n"
        "  - 100: 100 years threshold cuts.\n"
        "  - 1000: 1000 years threshold cuts.\n",
        default='100'
    )
    
    parser.add_argument(
        '--rbound',
        help="The max time considered for the threshold sliding"
    )
    
    parser.add_argument(
        '--lbound',
        help="The min time considered for the threshold sliding"
    )
    
    parser.add_argument('--context', help='Path to folder to store/retrieve application context [Will be created if does not exist]', required=True)
    
    parser.add_argument('--perf', action='store_true', help='Output plot of training\'s performance indicators [File perf_ind.png in running folder]')
    
    # Set tools from command line
    parser.add_argument('--PCA', action='store_true', help='Calculate PCA')
    parser.add_argument('--plot', action='store_true', help='Plot data')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')

    # Set output destionation from command line
    parser.add_argument(
        '--output', help='Set output destination [Default: .]', default='.')
    
    # Get arguments from command line
    args: Namespace = parser.parse_args()

    rbound = float(args.rbound)
    lbound = float(args.lbound)
    
    window = float(args.window)
    
    context_path = args.context
    
    subprocess.run(['mkdir', '-p', context_path + '/.tops/'])
    subprocess.run(['mkdir', '-p', context_path + '/models/'])
            
    model_name = args.model

    next_phase = ''
    if args.phase == 'multifasta':
        logging.info('Running multifasta phase...')
        
        subprocess.run(['rm', context_path + '/.tops/*'])
        
        read_multifasta()
        divide_set()
        extract_features()
        load_features()
        build_authenticator(model_name)

    elif args.phase == 'feature_extraction':
        logging.info('Running feature_extraction phase...')

        load_sets()
        extract_features()
        build_authenticator(model_name)
    
    elif args.phase == 'training':
        logging.info('Running training phase...')
        
        load_features()
        build_authenticator(model_name)
        
    elif args.phase == 'auth':
        pass
        
    elif args.verbose:
        verbose = True

    # if args.PCA:
    #     calculate_pca()
    #     return


    if args.plot:
        plot()


if __name__ == '__main__':
    main()

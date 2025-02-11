import re
import numpy as np
import joblib
import subprocess
import pandas as pd
import os
import logging
import authpipe.core.configuration as config
from authpipe.core.plot import plot_training_results, plot_binary_perf, plot_binary_auroc_auprc, plot_binary_hist_dist
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


def train(threshold, model_name):
    df_train = config.settings.df_train
    context_path = config.settings.context_path

    # Separate features (X) and target (y)
    X_train = df_train[['falcon_estimated_age',
                        'cg_content', 'relative_size', 'n_content']]
    # X_train = df_train[['falcon_estimated_age']]
    y_train = df_train['age']

    y_train = pd.DataFrame([1 if value >= threshold else 0 for value in y_train])
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
        logging.error(
            'Invalid model! Run \'authpipe --help\' to check the models available!')
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
    model_file_path = os.path.join(context_path, 'models/model_' + model_name + '_' + str(threshold) + '.joblib')
    joblib.dump(model, model_file_path)

    return acc, precision, recall, f1, y_train, y_pred, cm, y_train_categories


def test(threshold, model_name):
    df_test = config.settings.df_test
    df_train = config.settings.df_train
    context_path = config.settings.context_path

    # Separate features (X) and target (y)
    X_test = df_test[['falcon_estimated_age',
                      'cg_content', 'relative_size', 'n_content']]
    # X_test = df_test[['falcon_estimated_age']]
    y_test = df_test['age']
    # Save the trained model
    model_file_path = os.path.join(
        context_path, 'models/model_' + model_name + '_' + str(threshold) + '.joblib')
    # Load the saved model
    model = joblib.load(model_file_path)

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


def build_authenticator(model_name, plot_results):
    logging.verbose(f'Building model...')

    n_intervals = config.settings.n_intervals
    window = config.settings.window
    rbound = config.settings.rbound
    lbound = config.settings.lbound
    n_intervals_x = config.settings.n_intervals

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

    n_intervals = max_lim - min_lim + 1

    samples_per_century_array = {i: 0 for i in range(0, n_intervals)}

    for threshold in range(min_lim, max_lim):
        logging.verbose(f'Threshold cut: {threshold}/{n_intervals - 1}')

        train(
            threshold=window*threshold, 
            model_name=model_name
        )
        acc, precision, recall, f1, y_true, y_pred, cm, y_true_categories, auroc, auprc = test(
            threshold=window*threshold,
            model_name=model_name
        )

        # logging.debug(f'Accuracy: {acc}')

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
            i) for i in y_true_categories_unique if i < n_intervals}

        for key in samples_per_century.keys():
            samples_per_century_array[key] += samples_per_century[key]

    total_samples = 0
    for key in samples_per_century_array.keys():
        total_samples += samples_per_century_array[key]

    if total_samples == 0:
        logging.error(f'Zero samples found! Exiting...')
        exit()

    for key in samples_per_century_array.keys():
        samples_per_century_array[key] = samples_per_century_array[key]/total_samples

    true_predictions = [C00_array[i] + C11_array[i]
                        for i in range(len(C00_array))]
    false_predictions = [C01_array[i] + C10_array[i]
                         for i in range(len(C01_array))]

    if plot_results:
        logging.verbose(f'Plotting training results...')
        plot_training_results(acc_array, precision_array, recall_array, f1_weighted_array,
                              honest_coin_random_array, y_pred_ancient_array, false_predictions, random_array, auroc_array, auprc_array, samples_per_century_array, n_intervals_x)


def authentication():
    model = config.settings.model
    threshold = config.settings.threshold 
    context_path = config.settings.context_path 
    auth_path = config.settings.auth_path 
    df_auth = config.settings.df_auth
    
    model_file_path = os.path.join(
        context_path, 'models/model_' + model + '_' + str(threshold) + '.joblib')
    
    if not os.path.exists(model_file_path):
        logging.error(f'Model trained with threshold {threshold} does not exist! Reviewe your context/models/ folder to see the available models or train a new one')
        exit()

    model_instance = joblib.load(model_file_path)
    
    # Separate features (X) and target (y)
    X_test = df_auth[['falcon_estimated_age',
                      'cg_content', 'relative_size', 'n_content']]
    # X_test = df_auth[['falcon_estimated_age']]
    y_test = df_auth['id']
    
    y_pred = model_instance.predict(X_test)
    y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
    
    auth_results_file_path = os.path.join(auth_path, f'auth_results_{model}_{threshold}.txt')
    with open(auth_results_file_path, 'w') as auth_results_file:
        idx = 0
        for id in y_test:
            if y_pred[idx] == 1:
                result_line = f'Sample {id} classified as ANCIENT with threshold {threshold} and model {model}'
                auth_results_file.write(result_line)
                auth_results_file.write('\n')
                logging.verbose(result_line)
            else:
                result_line = f'Sample {id} classified as MODERN with threshold {threshold} and model {model}'
                auth_results_file.write(result_line)
                auth_results_file.write('\n')
                logging.verbose(result_line)

            idx += 1

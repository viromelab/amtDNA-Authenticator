import re
from argparse import ArgumentParser, Namespace
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from xgboost import XGBRegressor
import pandas as pd
from math import floor
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from colorama import Fore, Back, Style

LABEL_SIZE = 16
N_CENTURIES = 60

# Set the font size for tick labels
plt.rcParams['xtick.labelsize'] = LABEL_SIZE  # For x-axis tick labels
plt.rcParams['ytick.labelsize'] = LABEL_SIZE  # For y-axis tick labels

this_file_path = os.path.dirname(os.path.abspath(__file__))
context_path = None

df_train = None
df_val = None
df_test = None

verbose = False

samples = {}
samples_train = {}
samples_test = {}
samples_val = {}


def printWarningMessage(message):
    print(Fore.YELLOW + Style.BRIGHT + '[WARNING]', end=' ')
    print(message)
    print(Style.RESET_ALL)
    

def printErrorMessage(message):
    print(Fore.RED + Style.BRIGHT + '[ERROR]', end=' ')
    print(message)
    print(Style.RESET_ALL)


def printRunningMessage(message):
    print(Fore.BLUE + Style.BRIGHT + '> ', message)
    print(Style.RESET_ALL)


def sep_line():
    print('=' * 50)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Load data from multifasta


def read_multifasta(multifasta_file_path):
    global samples, this_file_path, N_CENTURIES

    multifasta_file = open(multifasta_file_path, 'r')
    multifasta = multifasta_file.read()
    multifasta = multifasta.upper()
    multifasta_file.close()

    pattern = re.compile(r'>(.*?)\n([\s\S]*?)(?=\n>|\Z)', re.DOTALL)

    fastas_content = pattern.findall(multifasta)

    max_age = 0
    
    for sample in fastas_content:
        try:
            raw_header = sample[0]
            header = raw_header.replace('>', '')
            header = header.split(' ')
            id = header[0]
            age = header[1]

            if not is_number(age):
                msg = 'Could not process age from sample with id ' + id + '. Found age: ' + age
                printWarningMessage()
                continue
            
            age = float(age)

            if age > max_age:
                max_age = age
                
            seq = sample[1]

            if id in samples:
                msg = 'Repeated sample with id ' + id
                printWarningMessage()

            samples[id] = [id, age, seq]
        except:
            msg = 'Could not process sample with id ' + id
            printErrorMessage(msg)
            
    N_CENTURIES = int(max_age/100)


def divide_set():
    global samples, samples_train, samples_test, samples_val

    train_file_path = os.path.join(context_path, 'trainset_index.txt')
    val_file_path = os.path.join(context_path, 'valset_index.txt')
    test_file_path = os.path.join(context_path, 'testset_index.txt')

    # Convert map to list
    samples_list = list(samples.items())
    # Shuffle list
    random.shuffle(samples_list)
    #
    samples_size = len(samples)
    #
    train_proportion = 0.7
    samples_train = samples_list[0:int(samples_size*train_proportion)]
    #
    val_proportion = 0.2
    samples_val = samples_list[int(
        samples_size*train_proportion):int(samples_size*(train_proportion+val_proportion))]
    #
    samples_test = samples_list[int(
        samples_size*(train_proportion+val_proportion)):]

    # Convert back to map
    samples = dict(samples_list)
    samples_train = dict(samples_train)
    samples_val = dict(samples_val)
    samples_test = dict(samples_test)

    with open(train_file_path, 'w') as file:
        for id in samples_train.keys():
            file.write(id)
            file.write('\n')
    with open(val_file_path, 'w') as file:
        for id in samples_val.keys():
            file.write(id)
            file.write('\n')
    with open(test_file_path, 'w') as file:
        for id in samples_test.keys():
            file.write(id)
            file.write('\n')


def load_sets():
    global samples, samples_train, samples_test, samples_val

    train_file_path = os.path.join(context_path, 'trainset_index.txt')
    val_file_path = os.path.join(context_path, 'valset_index.txt')
    test_file_path = os.path.join(context_path, 'testset_index.txt')

    if os.path.exists(train_file_path):
        with open(train_file_path, 'r') as file:
            train_ids = file.readlines()
            train_ids = [id.replace('\n', '') for id in train_ids]

            for id in train_ids:
                samples_train[id] = samples[id]

        with open(val_file_path, 'r') as file:
            val_ids = file.readlines()
            val_ids = [id.replace('\n', '') for id in val_ids]

            for id in val_ids:
                samples_val[id] = samples[id]

        with open(test_file_path, 'r') as file:
            test_ids = file.readlines()
            test_ids = [id.replace('\n', '') for id in test_ids]

            for id in test_ids:
                samples_test[id] = samples[id]

    else:
        msg = 'Could not find Train/Val/Test files to load. Try running again from phase \'multifasta\''
        printErrorMessage(msg)

## Calculate FALCON scores
def get_falcon_scores():
    global samples, samples_train, samples_test, samples_val

    # Build train multifasta
    train_multifasta_file_name = os.path.join(context_path, 'train_multifasta.fa')
    train_multifasta_file = open(train_multifasta_file_name, 'w')
    for key in samples_train.keys():
        sample = samples_train[key]
        id = sample[0]
        age = sample[1]
        header = '>' + id + ' ' + str(age)
        seq = sample[2].replace('\n', '')

        train_multifasta_file.write(header)
        train_multifasta_file.write('\n')
        train_multifasta_file.write(seq)
        train_multifasta_file.write('\n')

    train_multifasta_file.close()

    # Run on Falcon
    count = 1
    samples_len = len(samples.keys())
    for key in samples.keys():
        sample = samples[key]
        id = sample[0]
        age = sample[1]
        header = '>' + id  + ' ' + str(age)
        seq = sample[2]

        temp_fasta_file_name = os.path.join(context_path, 'temp_fasta.fa')
        temp_fasta_file = open(temp_fasta_file_name, 'w')
        temp_fasta_file.write(header)
        temp_fasta_file.write('\n')
        temp_fasta_file.write(seq)

        top_file_name = os.path.join(context_path, '.tops/' + id + '_' + str(age) + '_top.txt') 

        message = str(count) + '/' + str(samples_len)
        printRunningMessage(message)
        count += 1

        # ./FALCON -m 6:1:0:0/0 -m 11:10:0:0/0 -m 13:200:1:3/1  -g 0.85 -v -F -t 50 -n 6 -x top_n1.txt FASTA_SAMPLE MTDB
        subprocess.run(['FALCON', '-m', '6:1:0:0/0', '-m', '11:10:0:0/0', '-m', '13:200:1:3/1', '-g', '0.85', '-v', '-F', '-t', '50', '-n', '12', '-x', top_file_name, temp_fasta_file_name, train_multifasta_file_name])

        temp_fasta_file.close()


def integrate_falcon_data():
    df = pd.DataFrame(columns=['set', 'id', 'age', 'top'])

    folder_path = os.path.join(context_path,'.tops/')

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            header = filename.split('_')
            id = header[0]
            age = header[-2]

            top_path = os.path.join(folder_path, filename)
            file = open(top_path, '+r')
            
            top_lines = file.readlines()
            top_list = [item.replace('\n', '').split('\t') for item in top_lines]
            top_list = [[item[0], item[1], item[2], item[3].split('_')] for item in top_list]
            top_list = [[int(item[0]), int(item[1]), float(item[2]), item[3][0], float(item[3][-1])] for item in top_list]

            df = df._append({'id': id, 'age': age, 'top': top_list}, ignore_index=True)

    output_file_name = os.path.join(context_path, 'falcon_integrated_data.csv')
    df.to_csv(output_file_name)


def get_falcon_predictions():
    input_file_name = os.path.join(context_path, 'falcon_integrated_data.csv')
    df = pd.read_csv(input_file_name)
    
    df['top'] = df['top'].apply(eval)

    results = []

    set_age_df = pd.DataFrame(results, columns=['id', 'age', 'avg_age'])

    for iter, row in df.iterrows():
        top_list = row['top']
        id = row['id']
        age = row['age']
        
        norm_val = sum([item[2] for item in top_list if item[3] != id])

        if norm_val == 0:
            print('[Zero norm val] Id:', id)
            continue

        avg_age = sum([item[2]*item[4] for item in top_list if item[3] != id])

        avg_age /= norm_val

        results.append([id, age, avg_age])

    temp_df = pd.DataFrame(results, columns=['id', 'age','avg_age'])
    set_age_df = pd.concat([set_age_df, temp_df], ignore_index=True)

    #######################################################################################

    falcon_features = []

    for id in set_age_df['id']:
        temp_df = set_age_df[set_age_df['id'] == id]
        
        # if id in visited_id:
        #     continue
        # visited_id[id] = True

        avg_age = 0
        count = 0
        for iter, row in temp_df.iterrows():
            avg_age += row['avg_age']
            count += 1

        if count != 1:
            avg_age /= count

        falcon_features.append([id, age, avg_age])

    falcon_features_df = pd.DataFrame(falcon_features, columns=['id', 'age', 'avg_age'])
    
    # Save dataframe
    output_file_name = os.path.join(context_path, 'falcon_predictions.csv')
    falcon_features_df.to_csv(output_file_name, index=False)


def get_quantitative_data():
    seq_base_size = 17000
    sample_array = []

    for id in samples:  
        item = samples[id]

        id = item[0]

        id = id.split('_')[0]

        age = item[1]

        seq = item[2]
        seq = seq.replace('\n', '')
        
        # Capitalize all chars in string
        seq = seq.upper()

        seq_len = len(seq)

        # Extract relative size
        relative_size = seq_len / seq_base_size
        # Extract CG content
        cg_content = (seq.count('C') + seq.count('G')) / seq_len
        # Extract N content
        n_content = seq.count('N') / seq_len

        sample_array.append((id, age, relative_size, cg_content, n_content))
        
    sample_array.sort(key=lambda item: item[0])

    id_array = [i[0] for i in sample_array]
    age_array = [i[1] for i in sample_array]
    relative_size_array = [i[2] for i in sample_array]
    cg_content_array = [i[3] for i in sample_array]
    n_content_array = [i[4] for i in sample_array]

    # save modern data to csv file
    df = pd.DataFrame({'id': id_array, 'real_age': age_array, 'relative_size': relative_size_array, 'cg_content': cg_content_array, 'n_content': n_content_array})
    output_file_name = os.path.join(context_path, 'quantitative_data.csv')
    df.to_csv(output_file_name, index=False)


def merge_data():
    falcon_features_path = os.path.join(context_path, 'falcon_predictions.csv')
    df_falcon_features = pd.read_csv(falcon_features_path)
    q_features_path = os.path.join(context_path, 'quantitative_data.csv')
    df_q_features = pd.read_csv(q_features_path)

    samples_sets = [samples_train, samples_val, samples_test]
    samples_sets_names = ['train', 'val', 'test']

    idx = 0
    for sample_set in samples_sets:
        df_set = pd.DataFrame(columns=['id', 'age', 'falcon_estimated_age', 'relative_size', 'cg_content', 'n_content'])
        for id in sample_set:
            age = sample_set[id][1]

            id = id.split('_')[0]

            falcon_features = df_falcon_features[df_falcon_features['id']==id]

            try:
                falcon_estimated_age = falcon_features['avg_age'].values[0]
            except:
                falcon_estimated_age = falcon_features['avg_age']

            q_features = df_q_features[df_q_features['id']==id]

            try:
                cg_content = q_features['cg_content'].values[0]
            except:
                cg_content = q_features['cg_content']

            try:
                relative_size = q_features['relative_size'].values[0]
            except:
                relative_size = q_features['relative_size']

            try:
                n_content = q_features['n_content'].values[0]
            except:
                n_content = q_features['n_content']

            df_set = df_set._append({'id': id, 'age': age, 'falcon_estimated_age': falcon_estimated_age, 'relative_size': relative_size, 'cg_content': cg_content, 'n_content': n_content}, ignore_index=True)
        
        df_set_path = os.path.join(context_path, samples_sets_names[idx] + '_data.csv')
        idx += 1
        df_set.to_csv(df_set_path, index=False)


def extract_features():
    get_falcon_scores()
    integrate_falcon_data()
    get_falcon_predictions()
    get_quantitative_data()

    merge_data()


def plot_binary_auroc_auprc(auroc_arr, auprc_arr):
    figure, axis = plt.subplots(1, 1)  # Adjust figure size if needed

    axis.plot(range(1, N_CENTURIES), auroc_arr, color='blue', label='AUROC')
    axis.plot(range(1, N_CENTURIES), auprc_arr, color='red', label='AUPRC')

    axis.set_xlabel('Generations Ago Threshold Cut')  # Add labels to the axis
    axis.set_ylabel('Score')
    axis.legend(loc='lower right')

    # Adjusts subplot params so that subplots fit in to the figure area.
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_binary_hist_dist(samples_per_century_array):
    figure, axis = plt.subplots(1, 1)

    keys = list(samples_per_century_array.keys())
    values = list(samples_per_century_array.values())

    axis.bar(keys, values, alpha=0.5)

    axis.set_xlabel('Generations Ago')
    axis.set_ylabel('Normalized Percentage of Samples %')

    plt.show()
    plt.close()


def plot_binary_perf(acc, precision, recall, f1, y_true_random, y_pred_random, false_predictions, random_array):
    figure, axis = plt.subplots(1, 1)

    axis.plot(range(1, N_CENTURIES), acc, color='blue', label='Accuracy')
    axis.plot(range(1, N_CENTURIES), false_predictions,
              color='red', alpha=0.5, label='False Predictions')
    axis.plot(range(1, N_CENTURIES), random_array, color='green',
              linestyle='dashed', alpha=0.5, label='Random')
    axis.plot(range(1, N_CENTURIES), f1, color='black', label='F1 Score')

    axis.legend(loc='center right')

    axis.set_xlabel('Generations Ago Threshold Cut')
    axis.set_ylabel('Normalized Value %')

    plt.show()

    plt.close()


def plot():
    global train_fastas

    figure, axis = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

    sample_array_modern = []
    sample_array_ancient = []
    relative_size_array = []
    cg_content_array = []
    n_content_array = []
    age_array = []
    code_array = []

    seq_base_size = 17000

    cg_count = 0
    total_count = 0
    modern_count = 0
    ancient_count = 0
    rs_count = 0

    min_len = 0
    max_len = 0
    min_id = ''
    max_id = ''

    cond = False

    modern = False

    for item in train_fastas:
        header = item[0].replace('>', '')

        header = header.split(' ')

        id = header[0]
        age = header[1]
        code = header[2]

        seq = item[1]
        seq = seq.replace('\n', '')

        seq_len = len(seq)

        if 'Modern' in id:
            modern = True

        # if not modern:
            # continue

        total_count += 1

        if cond == False:
            min_len = seq_len
            max_len = seq_len
            min_id = id
            max_id = id
            cond = True

        elif seq_len < min_len:
            min_len = seq_len
            min_id = id

        elif seq_len > max_len:
            max_len = seq_len
            max_id = id

        # Extract relative size
        relative_size = seq_len / seq_base_size
        # Extract CG content
        cg_content = (seq.count('C') + seq.count('G')) / seq_len

        if cg_content > 0.3 and cg_content < 0.5:
            cg_count += 1

        if relative_size < 0.9748 and relative_size > 0.9:
            rs_count += 1

        # Extract N content
        n_content = seq.count('N') / seq_len

        if modern:
            modern_count += 1
            sample_array_modern.append(
                (id, code, age, relative_size, cg_content, n_content))
        else:
            ancient_count += 1
            sample_array_ancient.append(
                (id, code, age, relative_size, cg_content, n_content))

        modern = False

    for item in sample_array_ancient:
        code = float(item[1])
        age = float(item[2])

        if abs(100*code - age) > 100:
            print('Wrong code!')
            exit()

    for item in sample_array_modern:
        code = float(item[1])
        age = float(item[2])

        if abs(100*code - age) > 100:
            print('Wrong code!')
            exit()

    sample_array_modern.sort(key=lambda item: item[0])
    sample_array_ancient.sort(key=lambda item: item[0])

    id_array = [i[0] for i in sample_array_modern]
    code_array = [i[1] for i in sample_array_modern]
    age_array = [i[2] for i in sample_array_modern]
    relative_size_array = [i[3] for i in sample_array_modern]
    cg_content_array = [i[4] for i in sample_array_modern]
    n_content_array = [i[5] for i in sample_array_modern]

    axis.plot(code_array, cg_content_array, relative_size_array, color='blue')

    id_array = [i[0] for i in sample_array_ancient]
    code_array = [i[1] for i in sample_array_ancient]
    age_array = [i[2] for i in sample_array_ancient]
    relative_size_array = [i[3] for i in sample_array_ancient]
    cg_content_array = [i[4] for i in sample_array_ancient]
    n_content_array = [i[5] for i in sample_array_ancient]

    axis.plot(code_array, cg_content_array, relative_size_array, color='red')

    plt.show()


def load_features():
    global df_train, df_val, df_test

    df_train_path = os.path.join(context_path, 'trainset_features.csv')
    df_train = pd.read_csv(df_train_path)

    df_val_path = os.path.join(context_path, 'valset_features.csv')
    df_val = pd.read_csv(df_val_path)

    df_test_path = os.path.join(context_path, 'testset_features.csv')
    df_test = pd.read_csv(df_test_path)


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
        print('Invalid model!')
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


def train_authenticator(model_name):
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
    samples_per_century_array = {i: 0 for i in range(0, N_CENTURIES)}

    for threshold in range(1, N_CENTURIES):
        if not verbose:
            print(f'\r{threshold}/{N_CENTURIES}')

        train(threshold=100*threshold, model_name=model_name)
        acc, precision, recall, f1, y_true, y_pred, cm, y_true_categories, auroc, auprc = test(
                threshold=100*threshold, model_name=model_name)

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
            i) for i in y_true_categories_unique if i < N_CENTURIES}

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
    global samples, verbose, context_path

    parser = ArgumentParser()

    # Set input args
    # parser.add_argument('--seq2', help='Input aDNA file 2 [Optional]', default=None)

    #
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
    
    parser.add_argument('--context', help='Path to folder to store/retrieve application context [Will be created if does not exist]', required=True)

    parser.add_argument('--perf', action='store_true', help='Output plot of training\'s performance indicators [File perf_ind.png in running folder]')
    
    # Set tools from command line
    parser.add_argument('--PCA', action='store_true', help='Calculate PCA')
    parser.add_argument('--plot', action='store_true', help='Plot data')
    parser.add_argument('--verbose', action='store_true', help='Print data')

    # Set output destionation from command line
    parser.add_argument(
        '--output', help='Set output destination [Default: .]', default='.')
    
    # Get arguments from command line
    args: Namespace = parser.parse_args()

    context_path = args.context
    
    subprocess.run(['mkdir', '-p', context_path + '/.tops/'])
    subprocess.run(['mkdir', '-p', context_path + '/models/'])
            
    model_name = args.model

    next_phase = ''
    if args.phase == 'multifasta':
        multifasta_file_path = args.divset
        read_multifasta(multifasta_file_path)
        divide_set()
        extract_features()
        next_phase = 'feature_extraction'
    
    if next_phase == 'feature_extraction' or args.phase == 'feature_extraction':
        load_sets()
        extract_features()
        next_phase = 'training'
    
    if next_phase == 'training' or args.phase == 'training':
        load_features()
        train_authenticator(model_name)
        
    if args.phase == 'auth':
        pass
        
    if args.verbose:
        verbose = True

    # if args.PCA:
    #     calculate_pca()
    #     return


    if args.plot:
        plot()


if __name__ == '__main__':
    main()

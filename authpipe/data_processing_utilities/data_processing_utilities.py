import pandas as pd
import re
import random
import os
import subprocess
import logging
import authpipe.configuration.configuration as config
from authpipe.general_utilities.general_utilities import is_number

samples_train = None 
samples_test = None 
samples_val = None

def read_multifasta():
    context_path = config.settings.context_path
    window = config.settings.window
    
    multifasta_file_path = os.path.join(context_path, 'multifasta.fa')
    multifasta_file = open(multifasta_file_path, 'r')
    multifasta = multifasta_file.read()
    multifasta = multifasta.upper()
    multifasta_file.close()

    pattern = re.compile(r'>(.*?)\n([\s\S]*?)(?=\n>|\Z)', re.DOTALL)

    fastas_content = pattern.findall(multifasta)

    max_age = 0
    
    samples = {} 
    
    for sample in fastas_content:
        raw_header = sample[0]
        header = raw_header.replace('>', '')
        header = header.split(' ')
        id = header[0]
        age = header[1]

        if not is_number(age):
            msg = 'Could not process age from sample with id ' + id + '. Found age: ' + age
            logging.info(msg)
            continue
        
        age = float(age)

        if age > max_age:
            max_age = age
            
        seq = sample[1].replace('\n', '')

        if id in samples:
            msg = 'Repeated sample with id ' + id
            logging.warning(msg)

        samples[id] = [id, age, seq]
            
    config.settings.samples = samples
    if config.settings.n_intervals is None:  
        config.settings.n_intervals = int(max_age/window)
    
    
def divide_set():
    samples = config.settings.samples
    samples_train = config.settings.samples_train
    samples_val = config.settings.samples_val
    samples_test = config.settings.samples_test
    context_path = config.settings.context_path

    train_file_path = os.path.join(context_path, 'train_multifasta.fa')
    val_file_path = os.path.join(context_path, 'val_multifasta.fa')
    test_file_path = os.path.join(context_path, 'test_multifasta.fa')

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
    samples_val = samples_list[int(samples_size*train_proportion):int(samples_size*(train_proportion+val_proportion))]
    #
    samples_test = samples_list[int(samples_size*(train_proportion+val_proportion)):]

    # Convert back to map
    samples = dict(samples_list)
    samples_train = dict(samples_train)
    samples_val = dict(samples_val)
    samples_test = dict(samples_test)

    with open(train_file_path, 'w') as file:
        for key in samples_train.keys():
            sample = samples_train[key]
            id = sample[0]
            age = sample[1]
            header = '>' + id + ' ' + str(age)
            seq = sample[2].replace('\n', '')

            file.write(header)
            file.write('\n')
            file.write(seq)
            file.write('\n')
            
    with open(val_file_path, 'w') as file:
        for key in samples_val.keys():
            sample = samples_val[key]
            id = sample[0]
            age = sample[1]
            header = '>' + id + ' ' + str(age)
            seq = sample[2].replace('\n', '')

            file.write(header)
            file.write('\n')
            file.write(seq)
            file.write('\n')
            
    with open(test_file_path, 'w') as file:
        for key in samples_test.keys():
            sample = samples_test[key]
            id = sample[0]
            age = sample[1]
            header = '>' + id + ' ' + str(age)
            seq = sample[2].replace('\n', '')

            file.write(header)
            file.write('\n')
            file.write(seq)
            file.write('\n')
            
    config.settings.samples_train = samples_train
    config.settings.samples_val = samples_val
    config.settings.samples_test = samples_test


def load_sets():
    context_path = config.settings.context_path
    samples = config.settings.samples
    samples_train = config.settings.samples_train
    samples_val = config.settings.samples_val
    samples_test = config.settings.samples_test

    train_file_path = os.path.join(context_path, 'train_multifasta.fa')
    val_file_path = os.path.join(context_path, 'val_multifasta.fa')
    test_file_path = os.path.join(context_path, 'test_multifasta.fa')

    max_age = 0

    if os.path.exists(train_file_path):
        with open(train_file_path, 'r') as file:
            pattern = re.compile(r'>(.*?)\n([\s\S]*?)(?=\n>|\Z)', re.DOTALL)

            train_multifasta = file.read()
            train_multifasta = train_multifasta.upper()
            fastas_content = pattern.findall(train_multifasta)
            
            for sample in fastas_content:
                    raw_header = sample[0]
                    
                    header = raw_header.replace('>', '')
                    header = header.split('_')
                    id = header[0]
                    age = header[-1]
                    
                    if not is_number(age):
                        msg = 'Could not process age from sample with id ' + id + '. Found age: ' + age
                        logging.warning(msg)
                        continue
                    
                    age = float(age)

                    if age > max_age:
                        max_age = age
                        
                    seq = sample[1].replace('\n', '')

                    if id in samples:
                        msg = 'Repeated sample with id ' + id
                        logging.warning(msg)

                    samples_train[id] = [id, age, seq]
    else:
        msg = 'Could not find Train FASTA files to load.'
        logging.error(msg)
        exit()
        
    if os.path.exists(val_file_path):
        with open(val_file_path, 'r') as file:
            pattern = re.compile(r'>(.*?)\n([\s\S]*?)(?=\n>|\Z)', re.DOTALL)

            val_multifasta = file.read()
            val_multifasta = val_multifasta.upper()
            fastas_content = pattern.findall(val_multifasta)
            
            for sample in fastas_content:
                    raw_header = sample[0]
                    
                    header = raw_header.replace('>', '')
                    header = header.split('_')
                    id = header[0]
                    age = header[-1]
                    
                    if not is_number(age):
                        msg = 'Could not process age from sample with id ' + id + '. Found age: ' + age
                        logging.warning(msg)
                        continue
                    
                    age = float(age)

                    if age > max_age:
                        max_age = age
                        
                    seq = sample[1].replace('\n', '')

                    if id in samples:
                        msg = 'Repeated sample with id ' + id
                        logging.warning(msg)

                    samples_val[id] = [id, age, seq]
    else:
        msg = 'Could not find Train FASTA files to load.'
        logging.error(msg)
        exit()
        
    if os.path.exists(test_file_path):
        with open(test_file_path, 'r') as file:
            pattern = re.compile(r'>(.*?)\n([\s\S]*?)(?=\n>|\Z)', re.DOTALL)

            test_multifasta = file.read()
            test_multifasta = test_multifasta.upper()
            fastas_content = pattern.findall(test_multifasta)
            
            for sample in fastas_content:
                raw_header = sample[0]
                header = raw_header.replace('>', '')
                header = header.split(' ')
                id = header[0]
                age = header[-1]

                if not is_number(age):
                    msg = 'Could not process age from sample with id ' + id + '. Found age: ' + age
                    logging.warning(msg)
                    continue
                
                age = float(age)

                if age > max_age:
                    max_age = age
                    
                seq = sample[1]

                if id in samples:
                    msg = 'Repeated sample with id ' + id
                    logging.warning(msg)

                samples_test[id] = [id, age, seq]

    else:
        msg = 'Could not find Test FASTA files to load.'
        logging.error(msg)
        exit()
        
    samples = {**samples_train, **samples_test}


def get_falcon_scores():
    context_path = config.settings.context_path
    samples = config.settings.samples
    samples_train = config.settings.samples_train
    samples_val = config.settings.samples_val
    samples_test = config.settings.samples_test
    
    context_path = config.settings.context_path
    verbose = config.settings.verbose
    
    train_multifasta_file_name = os.path.join(context_path, 'train_multifasta.fa')
    
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

        msg = '[FALCON] ' + str(count) + '/' + str(samples_len)
        if verbose:
            logging.verbose(msg)
        count += 1
        
        with open(os.devnull, 'w') as devnull:
            # ./FALCON -m 6:1:0:0/0 -m 11:10:0:0/0 -m 13:200:1:3/1  -g 0.85 -v -F -t 50 -n 6 -x top_n1.txt FASTA_SAMPLE MTDB
            subprocess.run(['FALCON', '-m', '6:1:0:0/0', '-m', '11:10:0:0/0', '-m', '13:200:1:3/1', '-g', '0.85', '-F', '-t', '50', '-n', '12', '-x', top_file_name, temp_fasta_file_name, train_multifasta_file_name], stdout=devnull, stderr=devnull)
    
            # subprocess.run(['FALCON', '-m', '6:1:0:0/0', '-m', '11:10:0:0/0', '-m', '13:200:1:3/1', '-g', '0.85', '-F', '-t', '50', '-n', '12', '-x', top_file_name, temp_fasta_file_name, train_multifasta_file_name])

        temp_fasta_file.close()


def integrate_falcon_data():
    context_path = config.settings.context_path
    
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
    context_path = config.settings.context_path
    
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
            msg = f'Project {id} has zero average similarity value! Take a look at its top!'
            logging.error(msg)
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
    samples = config.settings.samples
    context_path = config.settings.context_path
    
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
    context_path = config.settings.context_path
    samples_train = config.settings.samples_train
    samples_val = config.settings.samples_val
    samples_test = config.settings.samples_test
    
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
        
        df_set_path = os.path.join(context_path, samples_sets_names[idx] + 'set_features.csv')
        idx += 1
        df_set.to_csv(df_set_path, index=False)


def extract_features():
    get_falcon_scores()
    integrate_falcon_data()
    get_falcon_predictions()
    get_quantitative_data()
    merge_data()


def load_features():
    window = config.settings.window
    
    context_path = config.settings.context_path
    
    df_train_path = os.path.join(context_path, 'trainset_features.csv')
    df_train = pd.read_csv(df_train_path)

    df_val_path = os.path.join(context_path, 'valset_features.csv')
    df_val = pd.read_csv(df_val_path)

    df_test_path = os.path.join(context_path, 'testset_features.csv')
    df_test = pd.read_csv(df_test_path)
    
    max_age = max(df_train['age'].max(), df_test['age'].max())
    max_age = max(max_age, df_val['age'].max())

    if config.settings.n_intervals is None:  
        config.settings.n_intervals = int(max_age/window)
    
    config.settings.df_train = df_train
    config.settings.df_val = df_val
    config.settings.df_test = df_test


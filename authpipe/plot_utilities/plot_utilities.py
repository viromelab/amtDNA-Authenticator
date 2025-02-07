import matplotlib.pyplot as plt

LABEL_SIZE = 16

# Set the font size for tick labels
plt.rcParams['xtick.labelsize'] = LABEL_SIZE  # For x-axis tick labels
plt.rcParams['ytick.labelsize'] = LABEL_SIZE  # For y-axis tick labels

def plot_binary_auroc_auprc(auroc_arr, auprc_arr, n_intervals_x):
    figure, axis = plt.subplots(1, 1)  # Adjust figure size if needed
    
    plt.ylim(0, 1)
    
    axis.plot(range(1, n_intervals_x), auroc_arr, color='blue', label='AUROC')
    axis.plot(range(1, n_intervals_x), auprc_arr, color='red', linestyle='dotted', label='AUPRC')

    axis.set_xlabel('Generations Ago Threshold Cut')  # Add labels to the axis
    axis.set_ylabel('Score')
    axis.legend(loc='lower left')

    # Adjusts subplot params so that subplots fit in to the figure area.
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_binary_hist_dist(samples_per_century_array):
    figure, axis = plt.subplots(1, 1)
    
    plt.ylim(0, 1)

    keys = list(samples_per_century_array.keys())
    values = list(samples_per_century_array.values())

    axis.bar(keys, values, alpha=0.5)
    
    axis.set_xlabel('Generations Ago')
    axis.set_ylabel('Normalized Percentage of Samples %')

    plt.show()
    plt.close()


def plot_binary_perf(acc, precision, recall, f1, y_true_random, y_pred_random, false_predictions, random_array, n_intervals_x):
    figure, axis = plt.subplots(1, 1)
    
    plt.ylim(0, 1)
    

    axis.plot(range(1, n_intervals_x), acc, color='blue', label='Accuracy')
    axis.plot(range(1, n_intervals_x), false_predictions,
              color='red', alpha=0.5, label='False Predictions')
    axis.plot(range(1, n_intervals_x), random_array, color='green',
              linestyle='dashed', alpha=0.5, label='Random')
    axis.plot(range(1, n_intervals_x), f1, color='black', label='F1 Score')

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

    for item in sample_array_modern:
        code = float(item[1])
        age = float(item[2])

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
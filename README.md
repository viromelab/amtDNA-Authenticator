<img src="assets/ai_mt.png" alt="ai_mt" width="800" height="400">


# A Machine Learning Tool for Authentication of Human Mitochondrial Ancient DNA

This repository contains code for a feature based machine learning tool for authenticating ancient DNA samples. The pipeline uses FALCON scores and quantitative features (relative size, CG content, N content) extracted from the DNA sequences.

## Features

* **FALCON Scores:**  Utilizes FALCON to generate similarity scores between the input DNA sequence and a reference database of DNA sequences.
* **Quantitative Features:** Extracts features like relative size, CG content, and N content from the DNA sequences.
* **Machine Learning Models:** Trains and evaluates different machine learning models (XGBoost, KNN, Neural Network, SVM, Gaussian Naive Bayes) for ancient DNA authentication.
* **Binary Classification:** Predicts whether a sample is above or below a certain age threshold.
* **Performance Evaluation:**  Includes metrics like accuracy, precision, recall, F1-score, AUROC, and AUPRC.


## Requirements

* Python 3.x
* scikit-learn
* xgboost
* pandas
* matplotlib
* joblib
* FALCON (needs to be installed separately)

## Installation

```
git clone git@github.com:viromelab/amtDNA-Authenticator.git
cd amtDNA-Authenticator
pip install -r requirements.txt
pip install -e .
git clone https://github.com/cobilab/falcon.git
cd falcon/src/
cmake .
make
cp FALCON ../../
cd ../../
```

## Example

```
$ cd ./script/
$ chmod +x build_examples.sh
$ source build_examples.sh
$ cd ..
```

It will build the examples for each mode of the program in the output folder `examples`. After running, the examples will be ready to run. Therefore, you do not need to prepare data, it is provided in the example in the correct format.

## Preparing Data (Not needed if running example):

    - Create a `multifasta.fa` file containing ancient DNA sequences for training, validation and testing. Place it in a first directory. This will be your `context` directory.

    - Create a `multifasta.fa` file containing ancient DNA sequences for authentication. Place it in a second directory. This will be your `auth` directory.

    - DNA sequences headers must have the format `>ID_AGE`.

## Usage

You can jump to any step in the pipeline, since all data necessary to run in any mode is already available (e.g., data generated in step "2. Run the extract-features mode" that is needed to run step "3. Run the train mode" is already available in the context directory of the training example).

### **1. Run the processing-multifasta mode:**

    - This mode will process the multifasta in the context folder, capitalizing the characters to ensure uniformity, removing duplicate samples, and dividing the data into training, validation and testing sets.

    ```
    cd examples/example_process_multifasta/
    authpipe process-multifasta --context context
    cd ..
    ```

### **2. Run the extract-features mode:**

    This command will load the data processed in the processing-multifasta mode and stored in the context folder, processing it further to extract features for the input vector of the machine learning model.

    *Step 2 can take several hours.

    ```
    authpipe extract_features --context ./context
    ```

    - Flag --falcon_verbose (-f): Show FALCON verbose

### **3. Run the train mode:**

    - This command will train XGBoost model instances on the extracted features with thresholds from 100 to 6000 years old and window size of 100 years. At the end of the run, it will print the results.

    ```
    authpipe train --context ./context --lbound 100 --rbound 6000 --window 100 --model XGB -p
    ```

    - You can change the `--model` argument to use a different machine learning model.
    - The `--window`, `--rbound`, and `--lbound` arguments control the age thresholds used for binary classification.
    - Flag --plot_results [-p] Plot results from training phase

### **4. Run the authenticate mode:**

    - This command authenticates new amtDNA sequences in multi-FASTA format, classifying them as ANCIENT/MODERN with respect to a 2000 years threshold, based on the model instances trained in the previous step.

    ```
    authpipe authenticate --context ./context --auth_path ./auth --threshold 2000 --model XGB
    ```
    
    - You can choose the model and threshold accordingly to the existent in the `context/models` folder.

## Outputs

* The trained model is saved in the `models` directory inside the context directory passed in command line.
* Performance metrics and plots are generated using pyplot and can be saved locally.
* Authentication files are saved in the `PATH-TO-AUTH-DIRECTORY` folder.

## License

GPLv3 License

Copyright (c) [2025] [Denis Yamunaque]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
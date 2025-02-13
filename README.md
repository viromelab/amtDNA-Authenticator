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

## Installation steps

You will need python3 installed beforee proceeding!

If you follow the example guide with prepared data in the `Example` section, you can jump this step for now. If you later prepare a new environment for another project, it is wise to come back to check these steps.

### **0. Create a virtual environment (venv):**

```
$ sudo apt-get install python3-venv # Install venv if not previously installed
$ cd PATH_TO_YOUR_PROJECT
$ python3 -m venv .venv # It created your venv in the hidden directory .venv (Use ls -la to see it)
$ source .venv/bin/activate # Activate venv
```

Now you have a venv configured to work in without messing with your global python installation.

To check if it is correctly working, you can run:

```
$ which python3
```

And it should return PATH_TO_YOUR_PROJECT/.venv/bin/python3

```
$ which pip
```

Should return PATH_TO_YOUR_PROJECT/.venv/bin/pip

### **1. Install the required dependencies on your venv with pip:**

```
$ pip install -r requirements.txt
```

### **2. Install FALCON from the repository:**

You might need to first configure your github account with the correct public key permissions. For that, follow this link https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent.

After setting up your permissions, proceede to install FALCON from this link:

https://github.com/cobilab/falcon

After installing, add falcon to your bin.

```
$ cp PARENT_DIR/src/FALCON /usr/bin/
```

in which `PARENT_DIR` is the directory where you previously cloned falcon.

## Example

Run the script `script/build_examples.sh`. From the amtDNA-Authenticator folder:

```
$ cd ./script/
$ chmod +x build_examples.sh
$ source build_examples.sh
```

It will build the examples for each mode of the program in the output folder `examples`. After running, the virtual environment will be already activated and the examples will be ready to run. Therefore, you do not need to prepare data, it is provided in the example in the correct format.

## Usage

If you run the `build_example.sh` script, `PATH-TO-CONTEXT-DIRECTORY` will be the `examples/example_MODE` directory, with `MODE` being process_multifasta, extract_features, train or authenticate. In that case, the data will be already prepared and you can test the program in any mode with the corresponding step from 1 to 4.

### **0. Prepare Data:**
    * Create a `multifasta.fa` file containing ancient DNA sequences for training, validation and testing. Place it in a first directory. This will be your `context` directory.

    * Create a `multifasta.fa` file containing ancient DNA sequences for authentication. Place it in a second directory. This will be your `auth` directory.

    * DNA sequences headers must have the format `>ID_AGE`.

### **1. Run the processing-multifasta mode:**
    * This mode will process the multifasta in the context folder, capitalizing the characters to ensure uniformity, removing duplicate samples, and dividing the data into training, validation and testing sets.

    ```
    authpipe process-multifasta --context PATH-TO-CONTEXT-DIRECTORY
    ```
    * Replace `PATH-TO-CONTEXT-DIRECTORY` with the actual path.

### **2. Run the extract-features mode:**
    * This mode will load the data processed in the processing-multifasta mode and stored in the context folder, processing it further to extract features for the input vector of the machine learning model.

    ```
    authpipe extract_features --context PATH-TO-CONTEXT-DIRECTORY
    ```

    * Replace `PATH-TO-CONTEXT-DIRECTORY` with the actual path (It should be the same as the used in the process-multifasta mode!).
    * Flag --falcon_verbose (-f): Show FALCON verbose

### **3. Run the train mode:**
    * This mode will train a machine learning model (Selected in command line) on the extracted features. 

    ```
    authpipe train --context PATH-TO-CONTEXT-DIRECTORY --lbound L_BOUND --rbound R_BOUND --window WINDOW --model MODEL
    ```

    * You can change the `--model` argument to use a different machine learning model.
    * The `--window`, `--rbound`, and `--lbound` arguments control the age thresholds used for binary classification.
    * Flag --plot_results [-p] Plot results from training phase

### **4. Run the authenticate mode:**
    * This mode authenticates new amtDNA sequences in multi-FASTA format, classifying them as ANCIENT/MODERN based on the models trained in the train mode.

    ```
    authpipe authenticate --context PATH-TO-CONTEXT-DIRECTORY --auth_path PATH-TO-AUTH-DIRECTORY --threshold THRESHOLD --model MODEL
    ```
    * Replace `PATH-TO-AUTH-DIRECTORY` with the path to a folder with the multi-FASTA containing the samples to be authenticated.
    * You can choose the model and threshold accordingly to the existent in the PATH-TO-CONTEXT-DIRECTORY/models folder.

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
<img src="figures/ai_mt.png" alt="Alt Text" width="800" height="400">

# A Machine Learning Tool for Authentication of Human Mitochondrial Ancient DNA

This repository contains code for a feature based machine learning tool for predicting the age of ancient DNA samples. The pipeline uses FALCON scores and quantitative features (relative size, CG content, N content) extracted from the DNA sequences.

## Features

* **FALCON Scores:**  Utilizes FALCON to generate similarity scores between the input ancient DNA and a reference database of ancient DNA sequences.
* **Quantitative Features:** Extracts features like relative size, CG content, and N content from the DNA sequences.
* **Machine Learning Models:** Trains and evaluates different machine learning models (XGBoost, KNN, Neural Network, SVM, Gaussian Naive Bayes) for age prediction.
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
* colorama

## Usage

1. **Prepare Data:**
    * Create a `multifasta.fa` file containing all ancient DNA sequences with headers in the format `>ID_AGE`.
    * Place the `multifasta.fa` file in a directory. This will be your `context` directory.
2. **Run the pipeline:**
    ```bash
    python main.py --phase multifasta --model XGB --window 100 --rbound 10000 --lbound 0 --context /path/to/your/context/directory
    ```
    * Replace `/path/to/your/context/directory` with the actual path.
    * You can change the `--model` argument to use a different machine learning model.
    * The `--window`, `--rbound`, and `--lbound` arguments control the age thresholds used for binary classification.

## Phases

The pipeline can be run in different phases:

* **multifasta:**  Starts with the `multifasta.fa` file, divides the data into training and testing sets, extracts features, and trains the model.
* **feature_extraction:**  Assumes the data is already divided into training and testing sets and proceeds with feature extraction and model training.
* **training:** Assumes the features are already extracted and proceeds with model training.
* **auth:**  Authenticates a new FASTA sequence using a pre-trained model (not implemented in the provided code).

## Output

* The trained model is saved in the `models` directory inside the context directory passed in command line.
* Performance metrics and plots are generated using pyplot and can be saved locally.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License

Copyright (c) [Year] [Your Name]

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
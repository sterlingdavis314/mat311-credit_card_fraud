# MAT 311 Intro to Data Science Final Project Customer Churn Prediction

This repository contains code relating my final project for class. This README file contains documentation relating to the purpose, the layout, and  running the project.

## Purpose

The purpose of this project is to build a simple model that determines whether a customer will churn or not based on the given dataset located inside `./data/raw/train.csv`. The datasets have been found under the kaggle competition and all submissions have been made through kaggle.

## Project layout

```
.
├── main.py                 # Entry point that runs the entire pipeline
├── requirements.txt        # Python dependencies
├── data/
│   ├── processed/          # Created after running the pipeline
│   └── raw/
│       └── card_transdata.csv
├── notebooks/
│   └── credit_card_fraud_analysis.ipynb
└── src/
    ├── data/
    │   ├── load_data.py
    │   ├── preprocess.py
    │   └── split_data.py
    ├── features/
    │   └── build_features.py
    ├── models/
    │   ├── train_model.py
    │   ├── dumb_model.py
    │   └── knn_model.py
    ├── utils/
    │   └── helper_functions.py
    └── visualization/
        ├── eda.py
        └── performance.py
```

`main.py` imports the modules inside `src/` and executes them to reproduce the analysis and results. Jupyter notebooks are provided only for prototyping and exploration—they are **not** meant to be the main entry point of the project.

Some directories such as `data/external/`, `src/utils/` and `tests/` may be empty, but the folder structure is provided to illustrate how a complete project should look.

## Running the example

Install the dependencies and run the pipeline. You should use the versions of the dependencies as specified by the requirements file:

```bash
conda create -n credit_fraud --file requirements.txt
conda activate credit_fraud
python main.py
```

This will load the dataset, perform basic feature engineering, train a simple model and produce visualizations similar to those in the notebook.
The cleaned data will be written to `data/processed/` and all plots will be displayed interactively.

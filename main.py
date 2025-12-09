from src.data.load_data import load_dataset
from src.data.preprocess import clean_data
from src.visualization.eda import plot_eda
from src.models.train_model import split_data, plot_roc_curve, plot_all_roc_curve, output_metrics
from src.models.knn_model import train_knn_model
from src.models.dumb_model import train_dumb_model
from src.models.decision_tree_model import train_decision_tree_model, features, columns_to_encode, columns_to_zero_set, columns_to_parse_as_dates
from src.visualization.performance import (
    plot_confusion_matrices,
    plot_performance_comparison,
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


def main() -> None:
    print("---Loading data...")
    raw_df = pd.read_csv("./data/raw/train.csv")
    
    # Print shape of the raw dataset
    print(f"Raw dataset shape: {raw_df.shape}")

    print("---Cleaning data...")
    clean_df = clean_data(raw_df, encode_attr=columns_to_encode, replace_attr=columns_to_zero_set, date_cols=columns_to_parse_as_dates)

    print(f"Cleaned dataset shape: {clean_df.shape}")

    # print("---Creating EDA visuals...")
    # plot_eda(clean_df)

    print("---Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(clean_df)

    print("---Training models...")
    dt_model = train_decision_tree_model(X_train, y_train)
    knn_model = train_knn_model(X_train, y_train)
    dumb_model = train_dumb_model(X_train, y_train)

    print("---Evaluating on validation set...")
    y_val_pred_dt = dt_model.predict(X_val)
    y_val_pred_knn = knn_model.predict(X_val)
    y_val_pred_dumb = dumb_model.predict(X_val)

    models = [
        {
            "name": "Dumb Model",
            "actual": y_val_pred_dumb,
            "color": "Reds",
            "model": dumb_model
        },
        {
            "name": "KNN Model",
            "actual": y_val_pred_knn,
            "color": "Blues",
            "model": knn_model
        },
        {
            "name": "Decision Tree Model",
            "actual": y_val_pred_dt,
            "color": "Greens",
            "model": dt_model
        },
    ]
    
    print("---Plotting Confusion Matricies...")
    plot_confusion_matrices(y_val, models)

    print("---Plotting ROC Curves...")
    plot_all_roc_curve(X_test, y_test, models)

    print("---Printing Metrics...")
    output_metrics(X_test, y_test, models)


    print("Done.")


if __name__ == "__main__":
    main()

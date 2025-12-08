import os
import pandas as pd
import numpy as np
from .load_data import load_dataset

from ..utils.pandas_extensions import replace_known_zero, one_hot_encoding

def clean_data(dataframe: pd.DataFrame, drop: bool = True, encode_attr: dict = {}, replace_attr: list[str] = [], date_cols: list[str] = []) -> pd.DataFrame:
    processed_data = dataframe.copy() 

    for col in encode_attr.keys():
        processed_data = one_hot_encoding(processed_data, col, possible_values=encode_attr[col])

    # processed_data = one_hot_encoding(processed_data, "Subscription Type")
    # processed_data = one_hot_encoding(processed_data, "Contract Length")
    # processed_data = one_hot_encoding(processed_data, "Gender")
    # processed_data = one_hot_encoding(processed_data, "Customer Status", possible_values=["active", "inactive"])

    for col in replace_attr:
        replace_known_zero(processed_data, column=col)

    for col in date_cols:
        processed_data[f"{col}_month"] = processed_data[col].str[0:2]
        processed_data[f"{col}_day"] = processed_data[col].str[3:5]

    # processed_data["Last Payment Month"] = processed_data["Last Payment Date"].datetime

    if drop:
        processed_data = processed_data.dropna()

    return processed_data

if __name__ == "__main__":
    # Load the raw dataset
    raw = load_dataset("data/raw/card_transdata.csv")
    # Clean the dataset
    cleaned = clean_dataset(raw)
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Save the cleaned data
    processed_path = "data/processed/card_transdata_clean.csv"
    cleaned.to_csv(processed_path, index=False)
    print(f"Cleaned data saved to {processed_path}")

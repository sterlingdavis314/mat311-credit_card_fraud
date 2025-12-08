import pandas as pd
import numpy as np

def replace_known_zero(df: pd.DataFrame, column: str) -> None:
    df[column] = df[column].replace("none", 0)
    df[column] = df[column].replace(np.nan, 0)
    df[column] = df[column].astype("float")

def one_hot_encoding(dataframe: pd.DataFrame, column: str, possible_values: list[str] = []) -> pd.DataFrame:
    if (column not in dataframe.columns):
        raise AssertionError(f"Column {column} doesn't exist")
    
    nparr = dataframe[column].value_counts()
    
    for new_col in possible_values:
        new_series = dataframe[column] == new_col
        dataframe[f"{column}_{new_col}"] = new_series

    for new_col in nparr.index:
        new_series = dataframe[column] == new_col
        dataframe[f"{column}_{new_col}"] = new_series


    return dataframe
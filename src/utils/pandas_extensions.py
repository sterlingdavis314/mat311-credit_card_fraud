import pandas as pd

def one_hot_encoding(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    if (column not in dataframe.columns):
        raise AssertionError(f"Column {column} doesn't exist")
    
    print(dataframe[column].value_counts().to_numpy()[0])

    return dataframe
    
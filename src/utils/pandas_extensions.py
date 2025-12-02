import pandas as pd

def one_hot_encoding(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    if (column not in dataframe.columns):
        raise AssertionError(f"Column {column} doesn't exist")
    
    nparr = dataframe[column].value_counts().to_numpy()

    # print(type(nparr))

    return dataframe
    
import pandas as pd


def one_hot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    One-hot encode a categorical column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The name of the categorical column to be one-hot encoded.

    Returns:
    pd.DataFrame: A new DataFrame with the one-hot encoded columns.
    """
    return pd.get_dummies(df, columns=[column], prefix=column, dtype=int)


def dummy(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Create dummy variables for a categorical column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The name of the categorical column to create dummy variables for.

    Returns:
    pd.DataFrame: A new DataFrame with the dummy variables.
    """
    return pd.get_dummies(df, columns=[column], prefix=column, drop_first=True, dtype=int)


def ordinal_encode(df: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    """
    Ordinal encode a categorical column in a DataFrame based on a specified order.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The name of the categorical column to be ordinal encoded.
    order (list): A list specifying the order of the categories.

    Returns:
    pd.DataFrame: A new DataFrame with the ordinal encoded column.
    """
    df_copy = df.copy()
    mapping = {category: rank for rank, category in enumerate(order)}
    df_copy[column] = df_copy[column].map(mapping)
    if df_copy[column].isnull().any():
        raise ValueError(f"Some values in column '{column}' are not in the specified order: {order}")
    return df_copy


def standard_scale(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Standard scale numerical columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): A list of numerical column names to be standard scaled.

    Returns:
    pd.DataFrame: A new DataFrame with the standard scaled columns.
    """
    df_copy = df.copy()
    for col in columns:
        std_dev = df_copy[col].std()
        mean = df_copy[col].mean()
        if std_dev == 0:
            df_copy[col] = 0.0  # constant column, avoid divide by zero
            continue
        df_copy[col] = (df_copy[col] - mean) / std_dev
    return df_copy


def min_max_scale(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Min-max scale numerical columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): A list of numerical column names to be min-max scaled.

    Returns:
    pd.DataFrame: A new DataFrame with the min-max scaled columns.
    """
    df_copy = df.copy()
    for col in columns:
        low = df_copy[col].min()
        spread = df_copy[col].max() - low
        if spread == 0:
            df_copy[col] = 0.0  # constant column, avoid divide by zero
            continue
        df_copy[col] = (df_copy[col] - low) / spread
    return df_copy

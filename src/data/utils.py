import pandas as pd
from typing import Dict
from sklearn.preprocessing import LabelEncoder


def encode_label(
        df: pd.DataFrame,
        col_to_encode: str,
        col_encoded: str = 'label_encoded') -> pd.DataFrame:
    """
    Encode arbitary valued labels to zero-index, consecutive labels.
    """

    assert col_to_encode in df.columns, f'{col_to_encode} is not a df column.'

    label_encoder = LabelEncoder()
    df[col_encoded] = label_encoder.fit_transform(df[col_to_encode])

    return df


def group_labels(
        df: pd.DataFrame,
        label_col: str = 'label_group',
        id_col: str = 'posting_id') -> pd.DataFrame:
    """
    This group the dataframe by `label_col` and map it to each item in `id_col`
    """

    for col in [id_col, label_col]:
        assert col in df.columns, f'Column {col} is not in df'

    label_ground_dict = (
        df
        .groupby(label_col)
        [id_col]
        .apply(set)
        .to_dict()
    )

    df['ground_truth'] = df[label_col].map(label_ground_dict)

    return df


def generate_idx_to_col_map(
        df: pd.DataFrame, col: str = 'posting_id') -> Dict:
    """
    Generate row index to column value mapping as a dictionary
    """
    idx_to_col_map = df[col].to_dict()

    return idx_to_col_map

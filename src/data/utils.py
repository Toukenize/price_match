import pandas as pd
from typing import Dict
from sklearn.preprocessing import LabelEncoder
from src.config.constants import DATA_SPLIT_PATH


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


def get_train_val_data(fold_num, data_col='title'):
    """
    Data has been pre-split into 10 folds, thus
    `fold_num` ranges from 0 to 9, and the rows
    with `split` == `fold_num` will be used as the
    validation set.
    """
    assert fold_num < 10,\
        f'fold_num + 1 : {fold_num + 1} is > 10'

    out_cols = ['posting_id', data_col, 'label_group']

    df = pd.read_csv(DATA_SPLIT_PATH)

    val_fold = f'fold_{fold_num}'

    train_df = (
        df
        .query(f'split != "{val_fold}"')
        .reset_index(drop=True)
        .copy()
    )

    val_df = (
        df
        .query(f'split == "{val_fold}"')
        .reset_index(drop=True)
        .copy()
    )

    return train_df[out_cols], val_df[out_cols]


def get_train_data_only(data_col='title'):
    """
    Use all data as train set.
    """
    out_cols = ['posting_id', data_col, 'label_group']
    df = pd.read_csv(DATA_SPLIT_PATH)

    return df[out_cols]

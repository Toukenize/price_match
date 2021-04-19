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


def get_holdout_data(data_col='title'):

    cols_req = [data_col, 'label_group']
    df = pd.read_csv(DATA_SPLIT_PATH)
    df['val_set'] = False
    df.loc[df['split'] == 'holdout', 'val_set'] = True

    return df[cols_req + ['val_set', 'posting_id']]


def get_train_val_data(fold_num, total_splits=8, data_col='title'):
    """
    Note that the splits is GroupedKFold, thus the
    validation set is to be used for similarity search
    rather than as the validation set for model training.

    Also, since the validation is done by searching against
    the full set of data, the returned "val_data" is actually
    the full dataframe, with additional column indicating
    whether the data row is from validation or not.
    """

    assert fold_num < total_splits,\
        f'fold_num + 1 : {fold_num + 1} is > total_split : {total_splits}'
    assert total_splits in [4, 8], 'total_splits must be 4 or 8'

    holdout_split = ['holdout']
    cols_req = [data_col, 'label_group']
    df = pd.read_csv(DATA_SPLIT_PATH)

    if total_splits == 4:
        val_splits = [f'fold_{fold_num*2}', f'fold_{fold_num*2+1}']

    else:
        val_splits = [f'fold_{fold_num}']

    not_train_splits = val_splits + holdout_split

    train_df = df.loc[~df['split'].isin(not_train_splits)]

    df['val_set'] = False
    df.loc[df['split'].isin(val_splits), 'val_set'] = True

    return train_df[cols_req], df[cols_req + ['val_set', 'posting_id']]

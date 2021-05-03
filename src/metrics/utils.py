import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Set, List, Tuple, Dict


def row_wise_f1(s1: Set, s2: Set) -> float:

    assert type(s1) is set, 's1 must be a set.'
    assert type(s2) is set, 's2 must be a set.'

    intersect_len = len(s1.intersection(s2))

    score = 2 * intersect_len / (len(s1) + len(s2))

    return score


def faiss_knn_cosine(
        emb_arr: np.ndarray,
        query_idx: List,
        neighbors: int = 50,
        chunksize: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Note:
        The distance returned is actually similarity ranges from 0 to 1.
        1 means both items are exactly the same.
    """
    # Infer feature dimension from emb_arr
    dim = emb_arr.shape[1]

    # Cast embeddings to float32 (Faiss only support this)
    emb_arr = emb_arr.astype(np.float32)

    # Normalize the emb arr in place
    faiss.normalize_L2(emb_arr)

    # Initialize Faiss Index
    index = faiss.IndexFlatIP(dim)
    index.add(emb_arr)

    # Find neighbors

    chunks = np.ceil(len(query_idx) / chunksize).astype(int)

    for i in tqdm(range(chunks), total=chunks, desc='>> Finding Neighbours'):

        query_idx_sub = query_idx[i*chunksize:(i+1)*chunksize]

        chunk_dists, chunk_indices = index.search(
            emb_arr[query_idx_sub], neighbors)

        if i == 0:
            dists = chunk_dists
            indices = chunk_indices
        else:
            dists = np.row_stack([dists, chunk_dists])
            indices = np.row_stack([indices, chunk_indices])

    return dists, indices


def faiss_knn_euclidean(
        emb_arr: np.ndarray,
        query_idx: List,
        neighbors: int = 50,
        chunksize: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Note:
        The euclidean distance is being measure here. 0 means both
        items are exactly the same.
    """
    # Infer feature dimension from emb_arr
    dim = emb_arr.shape[1]

    # Cast embeddings to float32 (Faiss only support this)
    emb_arr = emb_arr.astype(np.float32)

    # Initialize Faiss Index
    index = faiss.IndexFlatL2(dim)
    index.add(emb_arr)

    # Find neighbors

    chunks = np.ceil(len(query_idx) / chunksize).astype(int)

    for i in tqdm(range(chunks), total=chunks, desc='>> Finding Neighbours'):

        query_idx_sub = query_idx[i*chunksize:(i+1)*chunksize]

        chunk_dists, chunk_indices = index.search(
            emb_arr[query_idx_sub], neighbors)

        if i == 0:
            dists = chunk_dists
            indices = chunk_indices
        else:
            dists = np.row_stack([dists, chunk_dists])
            indices = np.row_stack([indices, chunk_indices])

    # Square root to get actual euclidean dist
    dists = np.sqrt(dists)

    return dists, indices


def faiss_knn_hamming(
        emb_arr: np.ndarray,
        query_idx: List,
        neighbors: int = 50,
        chunksize: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binary arr should be packed to their bytes equivalent, before
    the packed array is passed as `emb_arr`

    i.e.
        emb_arr = array([0, 0, 0, 0, 0, 0, 1, 0])
        packed_arr = np.packbits(emb_arr) # array([2], dtype=uint8)

    Note:
        The distance returned is actually similarity ranges from 0 to 1.
        1 means both items are exactly the same.
    """
    # Infer feature dimension from emb_arr
    dim = emb_arr.shape[1] * 8

    # Cast embeddings to float32 (Faiss only support this)
    emb_arr = emb_arr.astype(np.uint8)

    # Initialize Faiss Index
    index = faiss.IndexBinaryFlat(dim)
    index.add(emb_arr)

    # Find neighbors
    chunks = np.ceil(len(query_idx) / chunksize).astype(int)

    for i in tqdm(range(chunks), total=chunks, desc='>> Finding Neighbours'):

        query_idx_sub = query_idx[i*chunksize:(i+1)*chunksize]

        chunk_dists, chunk_indices = index.search(
            emb_arr[query_idx_sub], neighbors)

        if i == 0:
            dists = chunk_dists
            indices = chunk_indices
        else:
            dists = np.row_stack([dists, chunk_dists])
            indices = np.row_stack([indices, chunk_indices])

    # Change dist to similarity
    dists = 1 - (dists / dim)

    return dists, indices


def get_similar_items(
        df: pd.DataFrame, emb: np.ndarray, val_idx: List,
        idx_to_id_col_mapping: Dict, id_col: str = 'posting_id',
        metric: str = 'cosine',
        n: int = 50, chunksize: int = 512) -> pd.DataFrame:
    """
    Given a `df`, its embeddings `emb` (must be of the same length as `df`),
    and the `val_idx` (row index for validation, must correspond to the
    indices in `df`), find the number of closest `n` neighbors in batches
    using `chunksize`.

    `id_col` is the item id that is used to identify the neighbors, and
    `idx_to_id_col_mapping` is the row index to id mapping to map the nearest
    neighbors index (returned by faiss) to the actual item id.

    `metric` should be either cosine or hamming.

    After search, a similarity df `sim_df` is returned, which contains the
    row index, distance and item id of each row's top n neighbors.
    """
    assert len(emb) == len(df),\
        f'Num of emb {len(emb)} != Num of rows in df {len(df)}'
    assert type(val_idx) is list, 'val_idx must be a list'
    assert len(set(val_idx)) <= len(df),\
        f'Num of elements in val_idx {len(set(val_idx))}'\
        f' > df len {len(df)}'
    assert max(val_idx) < len(df),\
        f'{max(val_idx)} is out of range for'\
        f'df of shape {df.shape}'

    assert metric in ['cosine', 'hamming', 'euclidean'],\
        f'{metric} is invalid. Supported metrics : cosine, hamming, euclidean'

    if metric == 'cosine':
        distances, indices = faiss_knn_cosine(
            emb, val_idx, n, chunksize)
    elif metric == 'hamming':
        distances, indices = faiss_knn_hamming(
            emb, val_idx, n, chunksize)
    elif metric == 'euclidean':
        distances, indices = faiss_knn_euclidean(
            emb, val_idx, n, chunksize)

    sim_df_base = df.loc[val_idx, [id_col]].reset_index(drop=True).copy()
    sim_df_base['neighbors'] = val_idx

    if metric == 'euclidean':
        sim_df_base['distances'] = 0.0
    else:
        sim_df_base['distances'] = 1.0

    sim_df = sim_df_base.copy()

    sim_df['neighbors'] = pd.Series(indices.tolist())
    sim_df['distances'] = pd.Series(distances.tolist())

    # Exploding multiple columns of equal len & sequence
    # https://stackoverflow.com/a/59330040/10841164
    sim_df = (
        sim_df
        .set_index(id_col)
        .apply(pd.Series.explode)
        .reset_index()
    )

    # This step makes sure each item should at least be its own neighbour
    sim_df = (
        sim_df_base
        .append(sim_df)
        .drop_duplicates(subset=[id_col, 'neighbors'])
        .reset_index(drop=True)
    )

    sim_df['matches'] = sim_df['neighbors'].map(idx_to_id_col_mapping)

    return sim_df


def find_best_f1_score(
        sim_df: pd.DataFrame, truth_df: pd.DataFrame,
        thres_min: float = 0.2, thres_max: float = 0.9,
        thres_step: float = 0.02,
        more_than_thres: bool = True) -> Tuple[float, float]:
    """
    If Euclidean distance is used, `more_than_thres` should be set to False,
    such that neighbors are those with less than the threshold given.

    When hamming distance or cosine distance is used, `more_than_thres` should
    be set to True.
    """

    best_score, best_thres = -1., -1.

    for thres in tqdm(
            np.arange(thres_min, thres_max + (thres_step/10.), thres_step),
            desc='>> Finding Best Thres'):

        score = eval_score_at_thres(sim_df, truth_df, thres, more_than_thres)

        if score > best_score:
            best_score = score
            best_thres = thres

    return best_score, best_thres


def eval_score_at_thres(sim_df: pd.DataFrame, truth_df: pd.DataFrame,
                        thres: float, more_than_thres: bool = True) -> float:
    """
    Given `sim_df` which contains nearest neighbours info and distances
    and `truth_df` which contains the ground truth neighbours of items in
    `sim_df`, calculate the f1 score at threshold distance `thres`.

    If Euclidean distance is used, `more_than_thres` should be set to False,
    such that neighbors are those with less than the threshold given.

    When hamming distance or cosine distance is used, `more_than_thres` should
    be set to True.
    """

    unique_ids = sim_df['posting_id'].unique()
    num_ids = len(unique_ids)
    num_found = len(truth_df.loc[truth_df['posting_id'].isin(unique_ids)])

    assert num_found == num_ids,\
        f'num unique id in sim_df {num_ids} != num id in truth_df {num_found}'

    for col in ['distances', 'matches', 'posting_id']:
        assert col in sim_df.columns, f'Column `{col}` not in sim_df'

    for col in ['ground_truth', 'posting_id']:
        assert col in truth_df.columns, f'Column `{col}` not in truth_df'

    if more_than_thres:
        sel = (sim_df['distances'] > thres)
    else:
        sel = (sim_df['distances'] < thres)

    sim_items_sets = (
        sim_df
        .loc[sel]
        .groupby('posting_id')
        ['matches']
        .apply(lambda x: set(x))
        .to_dict()
    )

    truth_df = truth_df.copy()
    truth_df['matches'] = truth_df['posting_id'].map(sim_items_sets)
    truth_df['score'] = (
        truth_df
        .apply(lambda x: row_wise_f1(x.matches, x.ground_truth), axis=1)
    )

    score = truth_df['score'].mean()

    return score


def eval_top_k_accuracy(
        sim_df: pd.DataFrame, truth_df: pd.DataFrame,
        top_k: int = 1, ascending=False) -> float:

    sel = sim_df['posting_id'] != sim_df['matches']

    closest_df = (
        sim_df
        .loc[sel]
        .sort_values('distances', ascending=ascending)
        .groupby('posting_id', as_index=False)
        .nth(list(range(top_k)))
    )

    posting_label_map = (
        truth_df
        .set_index('posting_id')
        ['ground_truth']
        .to_dict()
    )

    correct = (
        closest_df
        .apply(
            lambda x: x.matches in posting_label_map[x.posting_id],
            axis=1)
        .sum()
    )

    accuracy = correct / len(truth_df)

    return accuracy

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Set, List, Tuple, Dict
from src.config.constants import KNN_CHUNKSIZE


def row_wise_f1(s1: Set, s2: Set) -> float:

    assert type(s1) is set, 's1 must be a set.'
    assert type(s2) is set, 's2 must be a set.'

    intersect_len = len(s1.intersection(s2))

    score = 2 * intersect_len / (len(s1) + len(s2))

    return score


def faiss_knn(
        emb_arr: np.ndarray,
        query_idx: List,
        neighbors: int = 50,
        chunksize: int = KNN_CHUNKSIZE) -> Tuple[np.ndarray, np.ndarray]:

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


def faiss_cos_dist(
        emb_arr: np.ndarray,
        query_idx: List,
        thresh: float = 0.99
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given an embedding array and a list of query_idx,
    return the incides and distances of the items with
    similarity threshold > thres.

    Refer to https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#range-search
    for more information on the outputs structure.
    """  # noqa

    # Infer feature dimension from emb_arr
    dim = emb_arr.shape[1]

    # Cast embeddings to float32 (Faiss only support this)
    emb_arr = emb_arr.astype(np.float32)

    # Normalize the emb arr in place
    faiss.normalize_L2(emb_arr)

    # Initialize Faiss Index
    index = faiss.IndexFlatIP(dim)
    index.add(emb_arr)

    # Search for neighbours with cos similarity > threshold
    lims, distances, indices = index.range_search(
        emb_arr[query_idx], thresh=thresh)

    return lims, distances, indices


def get_similar_items(
        df: pd.DataFrame, emb: np.ndarray, val_idx: List,
        idx_to_id_col_mapping: Dict, id_col: str = 'posting_id',
        n: int = 50, chunksize: int = 512) -> pd.DataFrame:
    """
    Given a `df`, its embeddings `emb` (must be of the same length as `df`),
    and the `val_idx` (row index for validation, must correspond to the
    indices in `df`), find the number of closest `n` neighbors in batches
    using `chunksize`.

    `id_col` is the item id that is used to identify the neighbors, and
    `idx_to_id_col_mapping` is the row index to id mapping to map the nearest
    neighbors index (returned by faiss) to the actual item id

    After search, a similarity df `sim_df` is returned, which contains the
    row index, distance and item id of each row's top n neighbors.
    """
    assert len(emb) == len(df),\
        f'Num of emb {len(emb)} != Num of rows in df {len(df)}'
    assert type(val_idx) is list, 'val_idx must be a list'
    assert len(set(val_idx)) < len(df),\
        f'Num of elements in val_idx {len(set(val_idx))}'\
        f' > df len {len(df)}'
    assert max(val_idx) < len(df),\
        f'{max(val_idx)} is out of range for'\
        f'df of shape {df.shape}'

    distances, indices = faiss_knn(
        emb, val_idx, n, chunksize)

    sim_df_base = df.loc[val_idx, [id_col]].reset_index(drop=True).copy()
    sim_df_base['neighbors'] = val_idx
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


def find_best_score(
        sim_df: pd.DataFrame, truth_df: pd.DataFrame,
        thres_min: float = 0.2, thres_max: float = 0.9,
        thres_step: float = 0.02) -> Tuple[float, float]:

    best_score, best_thres = -1., -1.

    for thres in tqdm(np.arange(thres_min, thres_max+0.01, thres_step),
                      desc='>> Finding Best Thres'):

        score = eval_score_at_thres(sim_df, truth_df, thres)

        if score > best_score:
            best_score = score
            best_thres = thres

    return best_score, best_thres


def eval_score_at_thres(sim_df: pd.DataFrame, truth_df: pd.DataFrame,
                        thres: float) -> float:
    """
    Given `sim_df` which contains nearest neighbours info and distances
    and `truth_df` which contains the ground truth neighbours of items in
    `sim_df`, calculate the f1 score at threshold distance `thres`.
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

    sel = (sim_df['distances'] > thres)

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

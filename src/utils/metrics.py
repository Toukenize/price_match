import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm


def row_wise_f1(s1, s2):

    assert type(s1) is set, 's1 must be a set.'
    assert type(s2) is set, 's2 must be a set.'

    intersect_len = len(s1.intersection(s2))

    score = 2 * intersect_len / (len(s1) + len(s2))

    return score


def faiss_knn(emb_arr, query_idx, neighbors=50, chunksize=512):

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

    for i in tqdm(range(chunks), total=chunks, desc='Finding Neighbours'):

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


def faiss_cos_dist(emb_arr, query_idx, thresh=0.99):
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


class KNNSearch:

    """
    This class does the KNN Search, given embeddings.
    """

    def __init__(self, df,
                 id_col='posting_id', label_col=None,
                 val_idx=None, eval_score=True, neighbors=50):

        # Check and init df
        assert id_col in df.columns, f'{id_col} not found in df'

        if eval_score:
            assert label_col is not None,\
                'label_col cannot be None when eval_score is True'
            assert label_col in df.columns,\
                f'{label_col} not found in df'

        self.id_col = id_col
        self.label_col = label_col
        self.df = df
        self.eval_score = eval_score

        # Check and init val_idx
        if val_idx is None:
            self.val_idx = self.df.index.to_list()
        else:
            assert type(val_idx) is list, 'val_idx can only be None or a list'
            assert len(set(val_idx)) < len(self.df),\
                f'Num of elements in val_idx {len(set(val_idx))}'\
                f' > df len {len(self.df)}'
            assert max(val_idx) < len(self.df),\
                f'{max(val_idx)} is out of range for'\
                f'df of shape {self.df.shape}'

            self.val_idx = val_idx

        self.neighbors = neighbors
        self.prepare_df()
        self.sim_df = None

    def prepare_df(self):

        if self.eval_score:
            label_ground_dict = (
                self.df
                .groupby(self.label_col)
                [self.id_col]
                .apply(set)
                .to_dict()
            )
            self.df['ground_truth'] = self.df[self.label_col].map(
                label_ground_dict)

        self.idx_to_posting_map = self.df[self.id_col].to_dict()

    def get_similar_items(self, emb):

        assert len(emb) == len(self.df),\
            f'Num of emb {len(emb)} != Num of rows in df {len(self.df)}'

        distances, indices = faiss_knn(emb, self.val_idx, self.neighbors)

        sim_df_base = self.df.loc[self.val_idx, [
            'posting_id']].reset_index(drop=True).copy()
        sim_df_base['neighbors'] = self.val_idx
        sim_df_base['distances'] = 1.0

        self.sim_df = sim_df_base.copy()

        self.sim_df['neighbors'] = pd.Series(indices.tolist())
        self.sim_df['distances'] = pd.Series(distances.tolist())

        # Exploding multiple columns of equal len & sequence
        # https://stackoverflow.com/a/59330040/10841164
        self.sim_df = (
            self.sim_df
            .set_index('posting_id')
            .apply(pd.Series.explode)
            .reset_index()
        )

        # This step makes sure each item should at least be its own neighbour
        self.sim_df = (
            sim_df_base
            .append(self.sim_df)
            .drop_duplicates(subset=['posting_id', 'neighbors'])
            .reset_index(drop=True)
        )

        self.sim_df['matches'] = self.sim_df['neighbors'].map(
            self.idx_to_posting_map)

    def eval_score_at_thres(self, thres):

        sel = (self.sim_df['distances'] > thres)

        sim_items_sets = (
            self.sim_df
            .loc[sel]
            .groupby('posting_id')
            ['matches']
            .apply(lambda x: set(x))
            .to_dict()
        )

        df_out = self.df.loc[self.val_idx].copy()

        df_out['matches'] = df_out['posting_id'].map(sim_items_sets)
        df_out['score'] = (
            df_out
            .apply(lambda x: row_wise_f1(x.matches, x.ground_truth), axis=1)
        )

        score = df_out['score'].mean()

        return score, df_out

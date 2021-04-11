import faiss
import numpy as np


def row_wise_f1(s1, s2):

    assert type(s1) is set, 's1 must be a set.'
    assert type(s2) is set, 's2 must be a set.'

    intersect_len = len(s1.intersection(s2))

    score = 2 * intersect_len / (len(s1) + len(s2))

    return score


def cos_dist_faiss(emb_arr, query_idx, thresh=0.99):
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

    TODO : Include pairwise distance in df or alt representation
    """

    def __init__(self, df,
                 id_col='posting_id', label_col=None,
                 val_idx=None, eval_score=True, thres=0.995):

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

        self.thres = thres
        self.prepare_df()

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

        lims, distances, indices = cos_dist_faiss(
            emb, self.val_idx, self.thres)

        preds = []

        # Note the that lims always have 1 more element than len(val_idx)
        for i in range(len(lims) - 1):

            sim_items = indices[lims[i]:lims[i + 1]]
            sim_items_set = set(
                self.idx_to_posting_map[item] for item in sim_items)
            preds.append(sim_items_set)

        self.df.loc[self.val_idx, 'matches'] = preds

        return self.df

    def evaluate_score_metric(self, emb):

        _ = self.get_similar_items(emb)

        self.df.loc[self.val_idx, 'score'] = (
            self.df
            .loc[self.val_idx]
            .apply(lambda x: row_wise_f1(x.matches, x.ground_truth), axis=1)
        )

        return self.df.loc[self.val_idx, 'score'].mean()

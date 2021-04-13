import os
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast
from src.data.utils import encode_label


class PriceMatchData(Dataset):

    def __init__(
            self, df,
            text_col='title',
            label_col=None,
            **tokenizer_args):

        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.label_col = label_col
        self.tokenizer = BertTokenizerFast.from_pretrained(**tokenizer_args)

        if self.label_col is not None:
            self.df = encode_label(
                self.df, col_to_encode=label_col, col_encoded=label_col)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        data = self.tokenizer(
            row[self.text_col],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        data = dict((k, v.squeeze()) for k, v in data.items())

        if self.label_col is not None:
            data['label'] = row[self.label_col]

        return data


def get_data_loader(
        df,
        text_col='title',
        label_col=None,
        shuffle=False,
        batch_size=32,
        **tokenizer_args):

    dataset = PriceMatchData(df, text_col, label_col, **tokenizer_args)
    dataloader = DataLoader(dataset, shuffle=shuffle,
                            batch_size=batch_size, num_workers=os.cpu_count())

    return dataloader

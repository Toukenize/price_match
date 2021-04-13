import pandas as pd
from src.data.nlp_dataset import get_data_loader
from src.config.constants import (
    DATA_SPLIT_PATH, PRETRAINED_NLP_MLM, PRETRAINED_TOKENIZER,
    NLP_CONFIG, SCHEDULER_MAPPING, OPTIMIZER_MAPPING, LOSS_MAPPING
)


def get_train_val_data(fold_num, total_splits=8):
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
    cols_req = ['title', 'label_group']
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


def get_model_optim_scheduler(model_class, num_classes, device):
    """
    Function to load model, optimizer and scheduler.

    Since the number of classes varies across splits,
    num_class needs to be provided.
    """
    model = model_class(
        PRETRAINED_NLP_MLM, num_classes, NLP_CONFIG['dropout_prob'])

    model = model.to(device)

    optimizer = OPTIMIZER_MAPPING[NLP_CONFIG["optimizer"]](
        params=model.parameters(),
        lr=NLP_CONFIG['learning_rate'])

    margin_loss = LOSS_MAPPING[NLP_CONFIG['loss_fn']](
        **NLP_CONFIG['loss_params'])

    scheduler_type = NLP_CONFIG['scheduler']

    if scheduler_type is not None:
        scheduler = SCHEDULER_MAPPING[scheduler_type](
            optimizer=optimizer,
            ** NLP_CONFIG['scheduler_params'])

    else:
        scheduler = None

    return model, optimizer, margin_loss, scheduler


def get_train_val_loaders(train_df, val_df):

    train_loader = get_data_loader(
        train_df, text_col='title', label_col='label_group',
        shuffle=True, batch_size=NLP_CONFIG['train_batch_size'],
        pretrained_model_name_or_path=PRETRAINED_TOKENIZER,
        model_max_length=NLP_CONFIG['model_max_length'])

    val_loader = get_data_loader(
        val_df, text_col='title', label_col=None,
        shuffle=False, batch_size=NLP_CONFIG['val_batch_size'],
        pretrained_model_name_or_path=PRETRAINED_TOKENIZER,
        model_max_length=NLP_CONFIG['model_max_length'])

    return train_loader, val_loader

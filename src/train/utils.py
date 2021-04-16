import pandas as pd
import torch
import torch.nn as nn

from src.data import nlp_dataset, img_dataset
from src.model.loss_func import ArcFace, CosFace
from src.config.base_model_config import ModelConfig, NLPConfig
from src.config.constants import DATA_SPLIT_PATH


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


def get_optimizer(config: ModelConfig, **kwargs):

    optim_name = config.optimizer

    if optim_name == 'adamw':
        optim_class = torch.optim.AdamW

    optimizer = optim_class(**kwargs)

    return optimizer


def get_scheduler(config: ModelConfig, **kwargs):

    schd_name = config.scheduler
    schd_params = config.scheduler_params

    if schd_name == 'reduce_on_plateau':
        schd_class = torch.optim.lr_scheduler.ReduceLROnPlateau
    elif schd_name == 'cyclic':
        schd_class = torch.optim.lr_scheduler.CyclicLR
    elif schd_name == 'cosine_anneal':
        schd_class = torch.optim.lr_scheduler.CosineAnnealingLR

    scheduler = schd_class(**schd_params, **kwargs)

    return scheduler


def get_loss_fn(config: ModelConfig):

    loss_fn_name = config.loss_fn
    loss_params = config.loss_params

    if loss_fn_name == 'arcface':
        loss_fn_class = ArcFace
    elif loss_fn_name == 'cosface':
        loss_fn_class = CosFace

    loss_fn = loss_fn_class(**loss_params)

    return loss_fn


def get_model_optim_scheduler(config: ModelConfig,
                              model_class: nn.Module, num_classes: int,
                              device: str):
    """
    Function to load model, optimizer and scheduler.

    Since the number of classes varies across splits,
    num_class needs to be provided.
    """

    torch.manual_seed(2021)  # For reproducibility

    model = model_class(
        config.pretrained_model_folder, num_classes, config.dropout_prob)

    model = model.to(device)

    optimizer = get_optimizer(
        config, params=model.parameters(), lr=config.learning_rate)

    margin_loss = get_loss_fn(config)

    scheduler = get_scheduler(config, optimizer=optimizer)

    return model, optimizer, margin_loss, scheduler


def get_nlp_train_val_loaders(
        nlp_config: NLPConfig,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame):

    train_loader = nlp_dataset.get_data_loader(
        train_df, text_col='title', label_col='label_group',
        shuffle=True, batch_size=nlp_config.train_batch_size,
        pretrained_model_name_or_path=nlp_config.pretrained_tokenizer_folder,
        model_max_length=nlp_config.model_max_length)

    val_loader = nlp_dataset.get_data_loader(
        val_df, text_col='title', label_col=None,
        shuffle=False, batch_size=nlp_config.val_batch_size,
        pretrained_model_name_or_path=nlp_config.pretrained_tokenizer_folder,
        model_max_length=nlp_config.model_max_length)

    return train_loader, val_loader


def get_img_train_val_loaders(
        img_config: ModelConfig,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame):

    train_loader = img_dataset.get_data_loader(
        train_df, text_col='title', label_col='label_group',
        shuffle=True, batch_size=img_config.train_batch_size)

    val_loader = img_dataset.get_data_loader(
        val_df, text_col='title', label_col=None,
        shuffle=False, batch_size=img_config.val_batch_size)

    return train_loader, val_loader

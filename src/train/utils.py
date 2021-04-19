import torch
import torch.nn as nn
from src.model.loss_func import ArcFace, CosFace
from src.config.base_model_config import ModelConfig


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


def get_optim_scheduler(config: ModelConfig,
                        model: nn.Module):
    """
    Function to load model, optimizer and scheduler.

    Since the number of classes varies across splits,
    num_class needs to be provided.
    """

    optimizer = get_optimizer(
        config, params=model.parameters(), lr=config.learning_rate)

    margin_loss = get_loss_fn(config)

    scheduler = get_scheduler(config, optimizer=optimizer)

    return optimizer, margin_loss, scheduler

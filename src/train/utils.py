import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup
from src.config.base_model_config import ModelConfig
from src.model.loss_func import ArcMarginProduct


def get_margin_func_and_params(config: ModelConfig):

    margin_func_name = config.loss_fn
    margin_params = config.loss_params

    if margin_func_name == 'arcmargin':
        margin_func = ArcMarginProduct

    return margin_func, margin_params


def get_optimizer(config: ModelConfig, **kwargs):

    optim_name = config.optimizer

    if optim_name == 'adamw':
        optim_class = torch.optim.AdamW

    optimizer = optim_class(**kwargs)

    return optimizer


def get_scheduler(config: ModelConfig, steps: int, **kwargs):

    schd_name = config.scheduler
    schd_params = config.scheduler_params

    if schd_name == 'reduce_on_plateau':
        schd_class = torch.optim.lr_scheduler.ReduceLROnPlateau

    elif schd_name == 'cosine_decay_w_warmup':
        schd_class = get_cosine_schedule_with_warmup
        schd_params = {
            'num_warmup_steps': schd_params['num_warmup_epochs'] * steps,
            'num_training_steps': schd_params['num_training_epochs'] * steps,
            'num_cyles': schd_params['num_cycles']
        }

    scheduler = schd_class(**schd_params, **kwargs)

    return scheduler


def get_optim_scheduler(config: ModelConfig,
                        model: nn.Module,
                        steps: int):
    """
    Function to load model, optimizer and scheduler.

    Since the number of classes varies across splits,
    num_class needs to be provided.
    """

    optimizer = get_optimizer(
        config, params=model.parameters(), lr=config.learning_rate)

    scheduler = get_scheduler(config, optimizer=optimizer, steps=steps)

    return optimizer, scheduler

import warnings
from pydantic import BaseModel, validator, DirectoryPath, FilePath
from typing import Dict, Union, Optional


class ModelConfig(BaseModel):

    epochs: int
    dropout_prob: float
    learning_rate: float
    train_batch_size: int
    val_batch_size: int
    scheduler: Union[str, None]
    scheduler_params: Optional[Dict]
    optimizer: str
    loss_fn: str
    loss_params: Optional[Dict]

    @validator('scheduler')
    def check_scheduler(cls, v):
        allowed = [None, 'cosine_decay_w_warmup']
        assert v in allowed, f'scheduler must be one of {allowed}'
        return v

    @validator('optimizer')
    def check_optimizer(cls, v):
        allowed = ['adamw']
        assert v in allowed, f'optimizer must be one of {allowed}'
        return v

    @validator('loss_fn')
    def check_loss_fn(cls, v):
        allowed = ['arcmargin']
        assert v in allowed, f'margin loss fn must be one of {allowed}'
        return v

    @validator('dropout_prob')
    def check_dropout(cls, v):
        if v > 0.5:
            warnings.warn(
                f"dropout_prob is {v} > 0.5, I hope you know what you're doing.")
        return v


class NLPConfig(ModelConfig):
    model_max_length: int
    pretrained_tokenizer_folder: DirectoryPath
    pretrained_model_folder: DirectoryPath


class IMGConfig(ModelConfig):
    img_dim: int
    pretrained_model_path: FilePath

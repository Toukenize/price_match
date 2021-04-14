from pathlib import Path
from torch.optim import lr_scheduler
from torch import optim
from src.model.loss_func import CosFace, ArcFace

# Data paths
DATA_FOLDER = Path('data/raw')
TRAIN_IMG_FOLDER = DATA_FOLDER / 'train_images'
DATA_SPLIT_PATH = DATA_FOLDER / 'train_split.csv'

# Pretrained model paths
MODEL_FOLDER = Path('model')
PRETRAINED_NLP_MLM = MODEL_FOLDER / 'indobert_lite_p2' / 'mlm_checkpoint'
PRETRAINED_TOKENIZER = MODEL_FOLDER / 'indobert_lite_p2' / 'tokenizer'

# Output paths
NLP_MODEL_PATH = MODEL_FOLDER / 'indobert_lite_p2' / 'emb_model_v2'

if not NLP_MODEL_PATH.exists():
    NLP_MODEL_PATH.mkdir()
else:
    # Lazy hack to avoid accidental overwritting of folder
    raise BaseException(f'Directory {NLP_MODEL_PATH} already exists.')

NLP_CONFIG = {
    "epochs": 3,
    "dropout_prob": 0.2,
    "model_max_length": 64,
    "learning_rate": 5e-5,
    "train_batch_size": 64,
    "val_batch_size": 128,
    "loss_fn": "arcface",
    "loss_params": {"m": 0.5, "s": 30.0},
    "optimizer": "adamw",
    "scheduler": "reduce_on_plateau",
    "scheduler_params": {"factor": 0.5, "patience": 2, "min_lr": 5e-6}
}

SCHEDULER_MAPPING = {
    'reduce_on_plateau': lr_scheduler.ReduceLROnPlateau,
    'cyclic': lr_scheduler.CyclicLR,
    'cosine_anneal': lr_scheduler.CosineAnnealingLR
}

OPTIMIZER_MAPPING = {
    'adamw': optim.AdamW
}

LOSS_MAPPING = {
    'arcface': ArcFace,
    'cosface': CosFace
}

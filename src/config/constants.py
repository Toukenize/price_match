import os
from pathlib import Path
from src.config.base_model_config import NLPConfig, IMGConfig

# Data paths
DATA_FOLDER = Path('data/raw')
IMG_FOLDER = Path('data/resize')
TRAIN_IMG_FOLDER = IMG_FOLDER / 'train_images_384'
DATA_SPLIT_PATH = DATA_FOLDER / 'train_split_v2.csv'

# Pretrained model paths
MODEL_FOLDER = Path('model')
PRETRAINED_NLP_MLM = MODEL_FOLDER / 'indobert_lite_p2' / 'pretrained'
PRETRAINED_TOKENIZER = MODEL_FOLDER / 'indobert_lite_p2' / 'tokenizer'
PRETRAINED_IMG = (MODEL_FOLDER / 'efficient_net_b0' /
                  'pretrained' / 'efficientnet_b0.pth')

# Output paths
NLP_MODEL_PATH = MODEL_FOLDER / 'indobert_lite_p2' / 'emb_model_v4'
IMG_MODEL_PATH = MODEL_FOLDER / 'efficient_net_b0' / 'emb_model_v1'

for path in [NLP_MODEL_PATH, IMG_MODEL_PATH]:
    if not path.exists():
        path.mkdir(parents=True)

# Dataloader Config
NUM_WORKER = os.cpu_count()

# KNN Chunksize
KNN_CHUNKSIZE = 1024

# NLP Configs
NLP_CONFIG = NLPConfig(
    epochs=50,
    dropout_prob=0.1,
    learning_rate=3e-4,
    train_batch_size=64,
    val_batch_size=128,
    scheduler='cosine_decay_w_warmup',
    scheduler_params={
        "num_warmup_epochs": 5,
        "num_training_epochs": 45,
        "num_cycles": 0.4},
    optimizer='adamw',
    loss_fn='arcmargin',
    loss_params={"m": 0.5, "s": 30.0, "easy_margin": False},
    model_max_length=48,
    pretrained_model_folder=PRETRAINED_NLP_MLM,
    pretrained_tokenizer_folder=PRETRAINED_TOKENIZER
)

# IMG Configs
IMG_CONFIG = IMGConfig(
    epochs=50,
    dropout_prob=0.1,
    learning_rate=8e-5,
    train_batch_size=16,
    val_batch_size=64,
    scheduler='cosine_decay_w_warmup',
    scheduler_params={
        "num_warmup_epochs": 5,
        "num_training_epochs": 45,
        "num_cycles": 0.4},
    optimizer='adamw',
    loss_fn='arcmargin',
    loss_params={"m": 0.5, "s": 30.0, "easy_margin": False},
    pretrained_model_path=PRETRAINED_IMG,
    img_dim=384
)

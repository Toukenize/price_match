from pathlib import Path
from src.config.base_model_config import NLPConfig, ModelConfig

# Data paths
DATA_FOLDER = Path('data/raw')
TRAIN_IMG_FOLDER = DATA_FOLDER / 'train_images'
DATA_SPLIT_PATH = DATA_FOLDER / 'train_split.csv'

# Pretrained model paths
MODEL_FOLDER = Path('model')
PRETRAINED_NLP_MLM = MODEL_FOLDER / 'indobert_lite_p2' / 'mlm_checkpoint'
PRETRAINED_TOKENIZER = MODEL_FOLDER / 'indobert_lite_p2' / 'tokenizer'
PRETRAINED_EFF_B3 = MODEL_FOLDER / 'efficient_net_b3'

# Output paths
NLP_MODEL_PATH = MODEL_FOLDER / 'indobert_lite_p2' / 'emb_model_test_refactor'
IMG_MODEL_PATH = MODEL_FOLDER / 'efficient_net_b3' / 'emb_model_v1'

for path in [NLP_MODEL_PATH, IMG_MODEL_PATH]:
    if not path.exists():
        path.mkdir(parents=True)

# KNN Chunksize
KNN_CHUNKSIZE = 512

# NLP Configs
NLP_CONFIG = NLPConfig(
    epochs=3,
    dropout_prob=0.2,
    learning_rate=3e-5,
    train_batch_size=32,
    val_batch_size=64,
    scheduler='reduce_on_plateau',
    scheduler_params={
        "factor": 0.5,
        "patience": 2,
        "min_lr": 5e-6},
    optimizer='adamw',
    loss_fn='arcface',
    loss_params={"m": 0.5, "s": 30.0},
    model_max_length=64,
    pretrained_model_folder=PRETRAINED_NLP_MLM,
    pretrained_tokenizer_folder=PRETRAINED_TOKENIZER
)

# IMG Configs
# IMG_CONFIG = ModelConfig(
#     epochs=100,
#     dropout_prob=0.2,
#     learning_rate=3e-5,
#     train_batch_size=64,
#     val_batch_size=512,
#     scheduler='reduce_on_plateau',
#     scheduler_params={
#         "factor": 0.5,
#         "patience": 2,
#         "min_lr": 5e-6},
#     optimizer='adamw',
#     loss_fn='arcface',
#     loss_params={"m": 0.5, "s": 30.0},
#     pretrained_model_folder=PRETRAINED_EFF_B3,
# )

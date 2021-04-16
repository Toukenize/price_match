import os
import torch
import argparse
import neptune.new as neptune
import numpy as np

from dotenv import load_dotenv
from src.model.img_model import ShopeeIMGModel
from src.train.loops import train, validate_w_knn
from src.train.utils import (
    get_img_train_val_loaders, get_train_val_data, get_optim_scheduler)
from src.config.constants import IMG_CONFIG, IMG_MODEL_PATH

# Init neptune logger
load_dotenv()
API_TOKEN = os.environ.get('NEPTUNE_TOKEN')
PROJ_NAME = os.environ.get('PROJECT_NAME')
RUN = neptune.init(project=PROJ_NAME, api_token=API_TOKEN,
                   tags=['IMG'], source_files='**/*.py',
                   description='Finetuning pretrained NLP with margin loss.')
RUN['params'] = IMG_CONFIG.dict()

# Init other training params
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = IMG_CONFIG.epochs


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--splits', choices=[4, 8], type=int, default=4,
        help="""
            Choose total number of cross validation splits (4 or 8).
            Default = 4
            """
    )
    parser.add_argument(
        '--val_freq', type=int, default=-1,
        help="""
            Validate and log F1-score at every `val_freq` number of epochs, in
            addition to the final epoch. If `val_freq` is negative, only
            validate at final epoch.
            Default = -1
            """
    )
    parser.add_argument(
        '--sample', type=float, default=1.0,
        help="""
            Specify the proportion of data to use in each fold (between 0 & 1).
            Default = 1.0
            """
    )
    parser.add_argument(
        '--trainfolds', nargs='+', type=int, default=[0, 1, 2, 3],
        help="""
            Specify the fold numbers to train (zero-index). Fold number cannot
            be larger than `split`
            Default = [0, 1, 2, 3]
            """
    )

    args = parser.parse_args()

    for f_num in args.trainfolds:
        assert f_num < args.splits,\
            f'trainfold {f_num + 1} > {args.splits} is invalid'

    return args


def main():

    # Get CLI arguments
    args = parse_args()
    splits = args.splits
    val_freq = args.val_freq
    sample = args.sample
    folds_to_train = args.trainfolds

    for fold_num in folds_to_train:

        # Init df based on split and fold number
        train_df, val_df = get_train_val_data(
            fold_num, splits, data_col='image')
        if sample != 1.0:
            train_df = train_df.sample(frac=sample, random_state=2021)
            val_df = val_df.sample(frac=sample, random_state=2021)

        # Init dataloaders and model-related stuff
        train_loader, val_loader = get_img_train_val_loaders(
            IMG_CONFIG, train_df, val_df)
        num_classes = train_df['label_group'].nunique()

        torch.manual_seed(2021)  # For reproducibility

        model = ShopeeIMGModel(
            IMG_CONFIG.pretrained_model_path, num_classes,
            IMG_CONFIG.dropout_prob)

        model = model.to(DEVICE)

        optimizer, margin_loss, scheduler = get_optim_scheduler(
            IMG_CONFIG, model)

        for epoch_num in range(EPOCHS):

            epoch_info = f'Fold {fold_num+1}, '\
                f'Epoch {epoch_num+1}/{EPOCHS}'

            # Compute & Log Train Losses
            train_losses = train(
                model, train_loader, optimizer, margin_loss,
                device=DEVICE, scheduler=scheduler, epoch_info=epoch_info)

            RUN[f'Fold_{fold_num + 1}_Train_Loss'].log(np.mean(train_losses))

            # Compute & Log F1 Score at val_freq / last epoch
            val_cond = (
                ((epoch_num + 1) % val_freq == 0)
                or
                (epoch_num + 1 == EPOCHS)
            )

            if val_cond:

                val_score, best_thres, sim_df, df = validate_w_knn(
                    model, val_loader, device=DEVICE)

                RUN['params/knn_thres'] = best_thres

                RUN[f'Fold_{fold_num + 1}_Val_F1_Score'].log(val_score)

        # Save model and last epoch pred
        torch.save(
            model,
            IMG_MODEL_PATH / f'fold_{fold_num + 1}_model.pt')

        sim_df.to_csv(
            IMG_MODEL_PATH / f'fold_{fold_num + 1}_pairwise_pred.csv',
            index=False)

        df.to_csv(
            IMG_MODEL_PATH / f'fold_{fold_num + 1}_grouped_pred.csv',
            index=False)

    return


if __name__ == "__main__":
    main()
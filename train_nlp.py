import os
import torch
import argparse
import neptune.new as neptune
import numpy as np

from dotenv import load_dotenv
from src.model.nlp_model import ShopeeNLPModel
from src.train.loops import train, validate_w_knn
from src.train.helper import (
    get_train_val_loaders, get_train_val_data, get_model_optim_scheduler)
from src.config.constants import NLP_CONFIG, NLP_MODEL_PATH

# Init neptune logger
load_dotenv()
API_TOKEN = os.environ.get('NEPTUNE_TOKEN')
PROJ_NAME = os.environ.get('PROJECT_NAME')
RUN = neptune.init(project=PROJ_NAME, api_token=API_TOKEN,
                   tags=['NLP'], source_files='**/*.py',
                   description='Finetuning pretrained NLP with margin loss.')
RUN['params'] = NLP_CONFIG

# Init other training params
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = NLP_CONFIG['epochs']


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
        '--thres', type=float, default=0.995,
        help="""
            Specify the cosine distance thres for validation (between 0 & 1).
            Default = 0.995
            """
    )
    args = parser.parse_args()

    return args


def main():

    # Get CLI arguments
    args = parse_args()
    splits = args.splits
    val_freq = args.val_freq
    sample = args.sample
    thres = args.thres

    RUN['params/knn_thres'] = thres

    for fold_num in range(splits):

        # Init df based on split and fold number
        train_df, val_df = get_train_val_data(fold_num, splits)
        if sample != 1.0:
            train_df = train_df.sample(frac=sample, random_state=2021)

        # Init dataloaders and model-related stuff
        train_loader, val_loader = get_train_val_loaders(train_df, val_df)
        num_classes = train_df['label_group'].nunique()
        model, optimizer, margin_loss, scheduler = get_model_optim_scheduler(
            ShopeeNLPModel, num_classes, device=DEVICE)

        for epoch_num in range(EPOCHS):

            epoch_info = f'Fold {fold_num+1}/{splits}, '\
                f'Epoch {epoch_num+1}/{EPOCHS}'

            # Compute & Log Train Losses
            train_losses = train(
                model, train_loader, optimizer, margin_loss,
                device=DEVICE, scheduler=scheduler, epoch_info=epoch_info)

            RUN[f'Fold_{fold_num + 1}_Train_Epoch'].log(epoch_num + 1)
            RUN[f'Fold_{fold_num + 1}_Train_Loss'].log(np.mean(train_losses))

            # Compute & Log F1 Score at val_freq / last epoch
            val_cond = (
                (
                    ((epoch_num % val_freq) == 0)
                    and
                    (epoch_num != 0)
                    and
                    (val_freq > 0)
                )
                or
                (epoch_num + 1 == EPOCHS))

            if val_cond:

                val_score, sim_df, df = validate_w_knn(
                    model, val_loader, thres, device=DEVICE)

                RUN[f'Fold_{fold_num + 1}_Val_Epoch'].log(epoch_num + 1)
                RUN[f'Fold_{fold_num + 1}_Val_F1_Score'].log(val_score)

        # Save model and last epoch pred
        torch.save(
            model.state_dict(),
            NLP_MODEL_PATH / f'fold_{fold_num + 1}_model.pt')

        sim_df.to_csv(
            NLP_MODEL_PATH / f'fold_{fold_num + 1}_pairwise_pred.csv',
            index=False)

        df.to_csv(
            NLP_MODEL_PATH / f'fold_{fold_num + 1}_grouped_pred.csv',
            index=False)

    return


if __name__ == "__main__":
    main()

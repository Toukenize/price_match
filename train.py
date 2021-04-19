import os
import torch
import argparse
import neptune.new as neptune
import numpy as np

from dotenv import load_dotenv
from src.train.loops import train_loop, validate_w_knn
from src.data.utils import get_train_val_data
from src.train.utils import get_optim_scheduler
from src.config.base_model_config import BaseModel
from src.config.constants import KNN_CHUNKSIZE


def get_neptune_run(config: BaseModel, model_tag: str, desc: str = ''):

    load_dotenv()
    api_token = os.environ.get('NEPTUNE_TOKEN')
    proj_name = os.environ.get('PROJECT_NAME')
    run = neptune.init(project=proj_name, api_token=api_token,
                       tags=[model_tag], source_files='**/*.py',
                       description=desc)
    run['params'] = config.dict()

    return run


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
    parser.add_argument(
        '--model_type', type=str, choices=['nlp', 'img'],
        help="""
            Specify type of training routine (nlp or img).
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
    model_type = args.model_type

    if model_type == 'img':

        from src.model.img_model import ShopeeIMGModel
        from src.data.utils import get_img_train_val_loaders
        from src.config.constants import IMG_CONFIG, IMG_MODEL_PATH

        config = IMG_CONFIG
        model_out_path = IMG_MODEL_PATH
        model_class = ShopeeIMGModel
        get_train_val_loaders = get_img_train_val_loaders
        data_col = 'image'

    elif model_type == 'nlp':
        from src.model.nlp_model import ShopeeNLPModel
        from src.data.utils import get_nlp_train_val_loaders
        from src.config.constants import NLP_CONFIG, NLP_MODEL_PATH

        config = NLP_CONFIG
        model_out_path = NLP_MODEL_PATH
        model_class = ShopeeNLPModel
        get_train_val_loaders = get_nlp_train_val_loaders
        data_col = 'title'

    # Init other stuff
    run = get_neptune_run(config, model_type.upper())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = config.epochs

    for fold_num in folds_to_train:

        # Init df based on split and fold number
        train_df, val_df = get_train_val_data(
            fold_num, splits, data_col=data_col)
        if sample != 1.0:
            train_df = train_df.sample(frac=sample, random_state=2021)
            val_df = val_df.sample(frac=sample, random_state=2021)

        # Init dataloaders and model-related stuff
        train_loader, val_loader = get_train_val_loaders(
            config, train_df, val_df)
        num_classes = train_df['label_group'].nunique()

        torch.manual_seed(2021)  # For reproducibility

        if model_type == 'img':
            model = model_class(
                config.pretrained_model_path, num_classes,
                config.dropout_prob)

        elif model_type == 'nlp':
            model = model_class(
                config.pretrained_model_folder, num_classes,
                config.dropout_prob)

        model = model.to(device)

        optimizer, margin_loss, scheduler = get_optim_scheduler(
            config, model)

        for epoch_num in range(epochs):

            epoch_info = f'Fold {fold_num+1}, '\
                f'Epoch {epoch_num+1}/{epochs}'

            # Compute & Log Train Losses
            train_losses = train_loop(
                model, train_loader, optimizer, margin_loss,
                device=device, scheduler=scheduler, epoch_info=epoch_info)

            run[f'Fold_{fold_num + 1}_Train_Loss'].log(np.mean(train_losses))

            # Compute & Log F1 Score at val_freq / last epoch
            val_cond = (
                ((epoch_num + 1) % val_freq == 0)
                or
                (epoch_num + 1 == epochs)
            )

            if val_cond:

                val_score, best_thres, sim_df = validate_w_knn(
                    model=model, dataloader=val_loader, device=device,
                    feature_dim=model.feature_dim, optimize=True,
                    n=50, chunksize=KNN_CHUNKSIZE)

                run['params/knn_thres'] = best_thres

                run[f'Fold_{fold_num + 1}_Val_F1_Score'].log(val_score)

        # Save model and last epoch pred
        torch.save(
            model,
            model_out_path / f'fold_{fold_num + 1}_model.pt')

        sim_df.to_csv(
            model_out_path / f'fold_{fold_num + 1}_pairwise_pred.csv',
            index=False)

    return


if __name__ == "__main__":
    main()

import os
import torch
import argparse
import neptune.new as neptune
import numpy as np

from src.train.loops import train_loop, validate_w_knn
from src.data.utils import get_train_val_data
from src.train.utils import get_optim_scheduler, get_margin_func_and_params
from src.config.base_model_config import BaseModel
from src.config.constants import KNN_CHUNKSIZE


def get_neptune_run(env: str,
                    config: BaseModel, model_tag: str, desc: str = ''):

    if env == 'local':
        from dotenv import load_dotenv
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
        '--env', type=str, default='local', choices=['local', 'kaggle'],
        help="""
            Specify training environment. Either local or kaggle.
            Default = local
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
        '--trainfolds', nargs='+', type=int, default=list(range(10)),
        help="""
            Specify the fold numbers to train (zero-index). Fold number cannot
            be larger than `split`
            Default = [0, 1, 2, 3, ... 9]
            """
    )
    parser.add_argument(
        '--model_type', type=str, default='nlp', choices=['nlp', 'img'],
        help="""
            Specify type of training routine (nlp or img).
            Default = nlp
            """
    )

    args = parser.parse_args()

    return args


def main():

    # Get var from CLI
    args = parse_args()
    env = args.env
    sample = args.sample
    folds_to_train = args.trainfolds
    model_type = args.model_type

    if model_type == 'img':

        from src.model.img_model import ShopeeIMGModel
        from src.data.img_dataset import get_img_train_val_loaders
        from src.config.constants import IMG_CONFIG, IMG_MODEL_PATH

        config = IMG_CONFIG
        model_out_path = IMG_MODEL_PATH
        model_class = ShopeeIMGModel
        get_train_val_loaders = get_img_train_val_loaders
        data_col = 'image'

    elif model_type == 'nlp':
        from src.model.nlp_model import ShopeeNLPModel
        from src.data.nlp_dataset import get_nlp_train_val_loaders
        from src.config.constants import NLP_CONFIG, NLP_MODEL_PATH

        config = NLP_CONFIG
        model_out_path = NLP_MODEL_PATH
        model_class = ShopeeNLPModel
        get_train_val_loaders = get_nlp_train_val_loaders
        data_col = 'title'

    # Init other stuff
    run = get_neptune_run(env, config, model_type.upper())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = config.epochs

    for fold_num in folds_to_train:

        # Init df based on split and fold number
        train_df, val_df = get_train_val_data(
            fold_num, data_col=data_col)
        if sample != 1.0:
            train_df = train_df.sample(frac=sample, random_state=2021)
            val_df = val_df.sample(frac=sample, random_state=2021)

        # Init dataloaders and model-related stuff
        train_loader, val_loader = get_train_val_loaders(
            config, train_df, val_df)
        num_classes = train_df['label_group'].nunique()

        torch.manual_seed(2021)  # For reproducibility

        margin_func, margin_params = get_margin_func_and_params(config)

        if model_type == 'img':
            model = model_class(
                config.pretrained_model_path,
                num_classes=num_classes,
                margin_func=margin_func,
                **margin_params)

        elif model_type == 'nlp':
            model = model_class(
                config.pretrained_model_folder,
                num_classes=num_classes,
                margin_func=margin_func,
                **margin_params)

        model = model.to(device)

        train_steps = len(train_loader)

        optimizer, scheduler = get_optim_scheduler(
            config, model, steps=train_steps)

        best_f1_score = 0.00
        best_train_loss = 10_000

        for epoch_num in range(epochs):

            epoch_info = f'Fold {fold_num+1}, '\
                f'Epoch {epoch_num+1}/{epochs}'

            # Compute & Log Train Losses
            train_losses = train_loop(
                model, train_loader, optimizer,
                device=device, scheduler=scheduler, epoch_info=epoch_info)

            avg_train_loss = np.mean(train_losses)

            run[f'Fold_{fold_num + 1}_Train_Loss'].log(avg_train_loss)

            val_acc, val_f1_score, f1_score_thres = validate_w_knn(
                model=model, dataloader=val_loader, device=device,
                feature_dim=model.feature_dim, optimize=True,
                n=50, chunksize=KNN_CHUNKSIZE)

            run[f'Fold_{fold_num + 1}_KNN_Thres'].log(f1_score_thres)
            run[f'Fold_{fold_num + 1}_Val_F1_Score'].log(val_f1_score)
            run[f'Fold_{fold_num + 1}_Val_Accuracy'].log(val_acc)

            # Save model if val_f1_score improved
            if val_f1_score > best_f1_score:
                model_name = f'fold_{fold_num + 1}_best_f1.pt'
                torch.save(
                    model.state_dict(),
                    model_out_path / model_name)
                best_f1_score = val_f1_score

            # Save model if best train loss improved
            if avg_train_loss < best_train_loss:
                model_name = f'fold_{fold_num + 1}_best_train_loss.pt'
                torch.save(
                    model.state_dict(),
                    model_out_path / model_name)
                best_train_loss = avg_train_loss

    return


if __name__ == "__main__":
    main()

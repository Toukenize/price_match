import gc
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.data.utils import generate_idx_to_col_map, group_labels
from src.metrics.utils import (
    get_similar_items, find_best_f1_score, eval_top_k_accuracy)


def train_loop(model, dataloader, optimizer, device, neptune_run=None,
               scheduler=None, epoch_info=''):

    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for i, data in pbar:

        data = dict((k, v.to(device)) for k, v in data.items())
        target = data.pop('label')

        # Compute output & loss
        logits = model.forward(label=target, **data)
        loss = criterion(logits, target)

        if scheduler is not None:
            lr = scheduler.get_lr()[0]
        else:
            lr = optimizer.param_groups[0]['lr']

        if neptune_run is not None:
            neptune_run[f'{epoch_info.split(",")[0]} LR'].log(lr)

        pbar.set_description(
            f'>> {epoch_info} - Train Loss : {loss.item():.4f} LR : {lr:.1e}')

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        losses.append(loss.item())

    return losses


def generate_embeddings(model, dataloader, device, feature_dim):

    model.eval()
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc='>> Generating embeddings'
    )
    batch_size = dataloader.batch_size
    emb_arr = np.zeros(
        (len(dataloader.dataset), feature_dim), dtype=np.float32)

    for i, data in pbar:

        data = dict((k, v.to(device)) for k, v in data.items())

        # Compute output & loss
        with torch.no_grad():
            features = model.extract_features(**data)

        features = features.cpu().numpy()

        emb_arr[i*batch_size:(i+1)*batch_size] = features

    return emb_arr


def validate_w_knn(model, dataloader, device,
                   feature_dim=768, optimize=True, **knn_params):

    emb_arr = generate_embeddings(model, dataloader, device, feature_dim)
    val_df = dataloader.dataset.df.copy()
    val_df = group_labels(val_df)
    idx_to_id_mapping = generate_idx_to_col_map(val_df)

    val_idx = val_df.index.tolist()

    sim_df = get_similar_items(
        val_df, emb_arr,
        val_idx, idx_to_id_mapping, **knn_params)

    # Clean up
    del emb_arr
    gc.collect()

    best_f1_score, best_f1_thres = find_best_f1_score(
        sim_df=sim_df, truth_df=val_df)

    acc_score = eval_top_k_accuracy(sim_df=sim_df, truth_df=val_df, top_k=1)

    return acc_score, best_f1_score, best_f1_thres

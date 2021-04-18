import gc
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.data.utils import generate_idx_to_col_map, group_labels
from src.metrics.utils import get_similar_items, find_best_score


def train_loop(model, dataloader, optimizer, margin_loss, device,
               scheduler=None, epoch_info=''):

    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for i, data in pbar:

        data = dict((k, v.to(device)) for k, v in data.items())
        target = data.pop('label')

        # Compute output & loss
        output = model.forward(**data)
        logits = margin_loss.forward(output, target)
        loss = criterion(logits, target)

        pbar.set_description(
            f'>> {epoch_info} (Train) - Margin Loss : {loss.item():.4f}')

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step(loss)

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
    val_idx = val_df.query('val_set == True').index.tolist()

    sim_df = get_similar_items(
        val_df, emb_arr,
        val_idx, idx_to_id_mapping, **knn_params)

    # Clean up
    del emb_arr
    gc.collect()

    if optimize:
        best_score, best_thres = find_best_score(
            sim_df=sim_df, truth_df=val_df.query('val_set == True'))
        return best_score, best_thres, sim_df

    else:
        return sim_df

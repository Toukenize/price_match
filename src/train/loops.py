import gc
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.metrics.utils import KNNSearch


def train(model, dataloader, optimizer, margin_loss, device, scheduler=None,
          epoch_info=''):

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


def validate_w_knn(model, dataloader, device,
                   thres_min=0.3, thres_max=0.8, thres_step=0.02):

    model.eval()
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc='>> Generating embeddings for validation'
    )
    batch_size = dataloader.batch_size
    hidden_size = model.hidden_size
    emb_arr = np.zeros(
        (len(dataloader.dataset), hidden_size), dtype=np.float32)

    for i, data in pbar:

        data = dict((k, v.to(device)) for k, v in data.items())

        # Compute output & loss
        with torch.no_grad():
            features = model.extract_features(**data)

        features = features.cpu().numpy()

        emb_arr[i*batch_size:(i+1)*batch_size] = features

    val_idx = dataloader.dataset.df.query('val_set == True').index.tolist()

    knn = KNNSearch(
        dataloader.dataset.df, label_col='label_group',
        val_idx=val_idx)

    knn.get_similar_items(emb_arr)

    best_score, best_thres = -1., -1.

    for thres in np.arange(thres_min, thres_max+0.01, thres_step):

        score, _ = knn.eval_score_at_thres(thres)

        if score > best_score:
            best_score = score
            best_thres = thres

    # Clean up
    del emb_arr
    gc.collect()

    return best_score, best_thres, knn.sim_df, knn.df

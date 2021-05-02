import cv2
import pandas as pd
import albumentations as a
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from src.data.utils import encode_label
from src.config.constants import IMGConfig, TRAIN_IMG_FOLDER, NUM_WORKER


class PriceMatchImgData(Dataset):
    def __init__(self, df, img_folder, transforms, img_path_col='image',
                 label_col=None):

        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.img_path_col = img_path_col
        self.label_col = label_col
        self.transforms = transforms

        if self.label_col is not None:
            self.df = encode_label(
                self.df, col_to_encode=label_col, col_encoded=label_col)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        img_path = self.img_folder / row[self.img_path_col]
        image = cv2.imread(img_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transforms(image=image)
        image = augmented['image']

        data = dict(image=image)

        if self.label_col is not None:
            data['label'] = row[self.label_col]

        return data


def get_train_transforms(img_dim):

    trans = a.Compose([
        a.Resize(img_dim, img_dim, always_apply=True),
        a.HorizontalFlip(p=0.5),
        a.VerticalFlip(p=0.5),
        a.Rotate(limit=120, p=0.8),
        a.RandomBrightness(limit=(0.09, 0.6), p=0.5),
        a.Normalize(),
        ToTensorV2(),
    ])

    return trans


def get_val_transforms(img_dim):

    trans = a.Compose([
        a.Resize(img_dim, img_dim, always_apply=True),
        a.Normalize(),
        ToTensorV2(),
    ])

    return trans


def get_data_loader(
        df,
        img_dim,
        img_folder,
        img_path_col='image',
        label_col=None,
        shuffle=False,
        batch_size=32):

    if label_col is None:
        transforms = get_val_transforms(img_dim)
    else:
        transforms = get_train_transforms(img_dim)

    dataset = PriceMatchImgData(
        df, img_folder, transforms, img_path_col, label_col)

    dataloader = DataLoader(dataset, shuffle=shuffle,
                            batch_size=batch_size, num_workers=NUM_WORKER,
                            pin_memory=True, drop_last=True)

    return dataloader


def get_img_train_val_loaders(
        img_config: IMGConfig,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        val_w_knn: bool = True
):

    train_loader = get_data_loader(
        train_df,
        img_dim=img_config.img_dim,
        img_folder=TRAIN_IMG_FOLDER,
        img_path_col='image', label_col='label_group',
        shuffle=True, batch_size=img_config.train_batch_size)

    if val_w_knn:
        val_label_col = None
    else:
        val_label_col = 'label_group'

    val_loader = get_data_loader(
        val_df, img_dim=img_config.img_dim,
        img_folder=TRAIN_IMG_FOLDER,
        img_path_col='image', label_col=val_label_col,
        shuffle=False, batch_size=img_config.val_batch_size)

    return train_loader, val_loader

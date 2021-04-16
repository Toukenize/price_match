import os
import cv2
import albumentations as a
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from src.data.utils import encode_label


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


def get_train_transforms(height, width):

    trans = a.Compose([
        a.Resize(height, width, always_apply=True),
        a.HorizontalFlip(p=0.5),
        a.VerticalFlip(p=0.5),
        a.Rotate(limit=120, p=0.8),
        a.RandomBrightness(limit=(0.09, 0.6), p=0.5),
        a.Normalize(),
        ToTensorV2(),
    ])

    return trans


def get_val_transforms(height, width):

    trans = a.Compose([
        a.Resize(height, width, always_apply=True),
        a.Normalize(),
        ToTensorV2(),
    ])

    return trans


def get_data_loader(
        df,
        img_height, img_width,
        img_folder,
        img_path_col='image',
        label_col=None,
        shuffle=False,
        batch_size=32):

    if label_col is None:
        print(f'label_col is {label_col}, getting VAL tranfroms')
        transforms = get_val_transforms(img_height, img_width)
    else:
        print(f'label_col is {label_col}, getting TRAIN tranfroms')
        transforms = get_train_transforms(img_height, img_width)

    dataset = PriceMatchImgData(
        df, img_folder, transforms, img_path_col, label_col)

    dataloader = DataLoader(dataset, shuffle=shuffle,
                            batch_size=batch_size, num_workers=os.cpu_count())

    return dataloader

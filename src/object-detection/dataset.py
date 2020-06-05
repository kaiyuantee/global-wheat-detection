# dataset
import pandas as pd
import numpy as np
import cv2
import ast
import torch
import random
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from .path import *


class GetDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        super().__init__()
        self.df = df
        self.image_ids = df['image_id'].unique()
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        if random.random() > 0.5:
            img, boxes = self.normal_load(index)
        else:
            img, boxes = self.cutmix_load(index)
        # area = boxes[:,-1]
        # area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        # iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes[:, :-1]
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        # target['area'] = area
        # target['iscrowd'] = iscrowd

        if self.transforms:
            for i in range(10):
                albu = self.transforms(**{'image': img,
                                          'bboxes': target['boxes'],
                                          'labels': labels})
                if len(albu['bboxes']) > 0:
                    # albu = self.transforms(**albu_params)
                    img = albu['image']  # output image after albumentations
                    target['boxes'] = torch.tensor(albu['bboxes'])
                    break

        return img, target, image_id

    def normal_load(self, index):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        img = cv2.imread(TRAIN_PATH + f'{image_id}.jpg', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # float32
        img = img / 255

        boxes = records[['xmin', 'ymin', 'xmax', 'ymax', 'area']].values
        return img, boxes

    def cutmix_load(self, index, imsize=1024):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)  # float32
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.normal_load(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        return result_image, result_boxes

    def __len__(self) -> int:
        return self.image_ids.shape[0]


def data_preprocessing(df):
    df[['xmin', 'ymin', 'width', 'height']] = pd.DataFrame([ast.literal_eval(x)
                                                            for x in df.bbox.tolist()], index=df.index)
    df['xmin'] = df['xmin'].astype(np.float)
    df['ymin'] = df['ymin'].astype(np.float)
    df['width'] = df['width'].astype(np.float)
    df['height'] = df['height'].astype(np.float)
    df['area'] = df['width'] * df['height']
    df['xmax'] = df['xmin'] + df['width']
    df['ymax'] = df['ymin'] + df['height']
    df = df[(df.area > 1000) & (df.area < 100000)]
    count = df.groupby('image_id').count().reset_index()
    veryfew = count[count.source < 4]
    df = df[~df.image_id.isin(veryfew.image_id.values)]
    return df


def train_valid_split(marking,fold):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df_folds = marking[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str))
    df_folds.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    df_train = marking[~marking.image_id.isin(df_folds[df_folds.fold == fold].index)]
    df_valid = marking[marking.image_id.isin(df_folds[df_folds.fold == fold].index)]
    return df_train, df_valid

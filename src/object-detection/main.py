import pandas as pd
import numpy as np
import argparse
import torch
import gc
import detection
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from .path import *
from .dataset import *
from .cfgs import *
from .transforms import *
from .utils import collate_fn
from .model import Fit
from detection.rpn import AnchorGenerator
from detection.faster_rcnn import FastRCNNPredictor
from detection.transform import GeneralizedRCNNTransform


def build_model(name: str, pretrained: bool, nms_threshold: float):  # threshold 0.25 pretrained: false
    anchor_sizes = [16, 32, 64, 128, 256]
    model = detection.__dict__[name](
        pretrained=pretrained,
        rpn_anchor_generator=AnchorGenerator(
            sizes=tuple((s,) for s in anchor_sizes),
            aspect_ratios=tuple((0.5, 1.0, 2.0) for _ in anchor_sizes),
        ),
        box_detections_per_img=200,
        box_nms_thresh=nms_threshold,
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=2)
    model.transform = ModelTransform(
        image_mean=model.transform.image_mean,
        image_std=model.transform.image_std,
    )
    return model


class ModelTransform(GeneralizedRCNNTransform):

    def __init__(self, image_mean, image_std):
        nn.Module.__init__(self)
        self.image_mean = image_mean
        self.image_std = image_std

    def resize(self, image, target):
        return image, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arg = parser.add_argument

    arg('--fold', default='0', help='folds')
    arg('--model', default='fasterrcnn_resnet_50_fpn', help='network')
    arg('--device', default='cuda', help='device')
    arg('--weights_path', default=None, help='weights path')
    arg('--resume', default=False, help='resume')
    arg('--resume_path', default=None, help='weight path')
    args = parser.parse_args()

    train = pd.read_csv(PATH + 'train.csv')
    train = data_preprocessing(train)
    df_train, df_valid = train_valid_split(train, args.fold)

    train_data = GetDataset(df_train,
                            TRAIN_PATH,
                            albu_train())
    val_data = GetDataset(df_valid,
                          TRAIN_PATH,
                          albu_val())

    train_data_loader = DataLoader(train_data,
                                   batch_size=Config.batch_size,
                                   sampler=RandomSampler(train_data),
                                   drop_last=True,
                                   num_workers=Config.n_workers,
                                   collate_fn=collate_fn)
    val_data_loader = DataLoader(val_data,
                                 batch_size=Config.batch_size,
                                 shuffle=False,
                                 sampler=SequentialSampler(val_data),
                                 num_workers=Config.n_workers,
                                 collate_fn=collate_fn)

    model = build_model(name=args.model, pretrained=False, nms_threshold=0.25)
    device = torch.device(args.device)
    model.to(device)

    gc.collect()
    Fit(model=model, config=Config, device=args.device, weights_path=args.weigths_path,
        resume_path=args.resume_path, resume=args.resume).fit(train_data_loader, val_data_loader)


if __name__ == '__main__':
    main()

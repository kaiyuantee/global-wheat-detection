# augmentation
import albumentations as A
from albumentations.pytorch import ToTensor


def albu_train():
    return A.Compose([A.RandomSizedBBoxSafeCrop(512, 512, p=1.0),
                      A.HorizontalFlip(p=0.5),
                      A.VerticalFlip(p=0.5),
                      A.OneOf([A.RandomGamma(),
                               A.RandomBrightnessContrast(brightness_limit=0.2,
                                                          contrast_limit=0.2),
                               A.HueSaturationValue(hue_shift_limit=0.2,
                                                    sat_shift_limit=0.2,
                                                    val_shift_limit=0.2)], p=0.9),
                      #A.CLAHE(p=1.0),  # CLAHE only supports uint8
                      A.MedianBlur(blur_limit=7, p=0.5),
                      A.CoarseDropout(max_height=64, max_width=64,
                                      fill_value=0, min_holes=2,
                                      min_height=8, min_width=8, p=0.5),

                      A.InvertImg(p=0.5),
                      ToTensor()],
                     p=1.0,
                     bbox_params={'format': 'pascal_voc',
                                  'min_area': 0,
                                  'min_visibility': 0,
                                  'label_fields': ['labels']})


def albu_val():
    return A.Compose([A.Resize(height=512, width=512, p=1.0),
                      A.HorizontalFlip(p=0.5),
                      A.VerticalFlip(p=0.5),
                      A.OneOf([A.RandomGamma(),
                               A.RandomBrightnessContrast(brightness_limit=0.2,
                                                          contrast_limit=0.2),
                               A.HueSaturationValue(hue_shift_limit=0.2,
                                                    sat_shift_limit=0.2,
                                                    val_shift_limit=0.2)], p=0.9),
                      #A.CLAHE(p=1.0),  # CLAHE only supports uint8
                      A.MedianBlur(blur_limit=7, p=0.5),
                      #                     A.CoarseDropout(max_height=64,max_width=64,
                      #                                     fill_value=0,min_holes=2,
                      #                                     min_height=8,min_width=8,p=0.5),

                      A.InvertImg(p=0.5),
                      ToTensor()],
                     p=1.0,
                     bbox_params={'format': 'pascal_voc',
                                  'min_area': 0,
                                  'min_visibility': 0,
                                  'label_fields': ['labels']})

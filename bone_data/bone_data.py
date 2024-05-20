#from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandCropByPosNegLabeld,
    ToTensord,
    AddChanneld,
    MapTransform,
    RandAdjustContrastd,
    SpatialPadd,
    Rand3DElasticd,
    RandAffined,


)   
from monai.data import DataLoader, Dataset
import json
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import Transform


#set_determinism(seed=2) # 555
with open('bone_data/data_bone.json') as f:
    data = json.load(f)
train_files, val_files, test_files = data['train'], data['val'], data['test']


def one_hot_to_class_map(one_hot_labels):
    """
    Convert one-hot encoded labels to class mapping.

    Args:
    - one_hot_labels (numpy.ndarray): One-hot encoded labels with shape (C, N, H, W),
                                      where C is the number of classes (9 in this case),
                                      N is the number of images,
                                      H is the height, and W is the width.

    Returns:
    - numpy.ndarray: Array with shape (N, H, W), where each pixel value represents the class index.
    """
    # Argmax operation along the class dimension (0) to get the class index for each pixel
    class_map = np.argmax(one_hot_labels, axis=0)
    return class_map

class LogShape(Transform):
    """
    Custom transform to log the shape of image and label tensors.
    """
    def __call__(self, data):
        for key in data.keys():
            if key in ['image', 'label']:
                print(f"{key} shape: {data[key].shape}")
        return data


class ReorderDims(Transform):
    """
    Reorder the dimensions of the image and label to ensure (Depth, Height, Width).
    """
    def __call__(self, data):
        d = dict(data)
        for key in ['image', 'label']:
            d[key] = d[key].transpose(-3, -1).transpose(-2, -1)  # Assuming original order is HWD
        return d



class ConvertToMultiChannelForBONEClassesd(MapTransform):

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 0)
            result.append(d[key] == 1)
            result.append(d[key] == 2)

            multi_channel_label = np.stack(result, axis=0).astype(np.float32)
            d[key] = multi_channel_label
            
        return d
        
def get_train_dataloader():

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ReorderDims(),
            ConvertToMultiChannelForBONEClassesd(keys = ['label']),
            AddChanneld(keys = ["image"]),
            SpatialPadd(keys=["image", "label"], spatial_size=(32, 320, 320)),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(32, 320, 320),
                pos = 0.6),   
            Rand3DElasticd(
                keys=["image", "label"],
                sigma_range=(9, 13),
                magnitude_range=(0, 900),
                prob=0.2,
                rotate_range=(0, 0, 0), 
                shear_range=None,
                translate_range=None,
                scale_range=None,
                mode=('bilinear', 'nearest'),
                padding_mode='border'
            ),
            RandAffined(
                keys=["image", "label"],
                prob=0.2,
                rotate_range=(0, 0, 0),
                scale_range=(0.85, 1.25),
                mode=('bilinear', 'nearest')
            ), 
            RandFlipd(keys = ["image", "label"],
                      prob = 0.5,
                      spatial_axis = 0),
             RandFlipd(keys = ["image", "label"],
                      prob = 0.5,
                      spatial_axis = 1),
            RandAdjustContrastd(
                keys = ["image"],
                gamma = (0.5, 4.5),
            ),
            RandScaleIntensityd(keys = "image", prob = 0.4, factors = 0.1),
            NormalizeIntensityd(keys = "image",
                                nonzero = True,
                                channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    train_ds = Dataset(data=train_files, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers = 4, 
                              pin_memory = True,  )


    return train_loader

def get_val_dataloader():
    #largest_size = (11, 256, 256)
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ReorderDims(),
            ConvertToMultiChannelForBONEClassesd(keys = ['label']),
            AddChanneld(keys = ["image"]),
            SpatialPadd(keys=["image", "label"], spatial_size=(32, 320, 320)),  
            RandScaleIntensityd(keys = "image", prob = 0.4, factors = 0.1),
            NormalizeIntensityd(keys = "image",
                               nonzero = True,
                               channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_ds = Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size = 1, shuffle=False, num_workers = 0,  
                            pin_memory = True)#drop_last = True

    return val_loader

def get_test_dataloader():
    test_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ReorderDims(),
            ConvertToMultiChannelForBONEClassesd(keys = ['label']),
            AddChanneld(keys = ["image"]),
            NormalizeIntensityd(keys = "image",
                               nonzero = True,
                               channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    test_ds = Dataset(data=test_files, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return test_loader

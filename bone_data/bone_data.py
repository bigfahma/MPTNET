from monai.utils import set_determinism
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
    Affined,
    Rand3DElasticd,
    RandAdjustContrastd,
    # RandShiftIntensityd,
    # Orientationd,
    # ScaleIntensityRanged,
    # SpatialPadd,
    # CropForegroundd,
    # RandSpatialCropSamplesd,
    

)   
from monai.data import DataLoader, Dataset
import json
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


set_determinism(seed=1)
with open('bone_data/data_bone_08_15_split.json') as f:
    data = json.load(f)
train_files, val_files, test_files = data['train'], data['val'], data['test']


class PrintShape:
    def __call__(self, data):
        print(f"Image shape: {data['image'].shape}")
        print(f"Label shape: {data['label'].shape}")
        return data

class DataViewImage:
    def __call__(self,data):
        print(np.array(data["image"][0]).shape)
        return data
class DataViewLabel:
    def __call__(self,data):
        pv.plot(np.array(data["label"][0]))
        pv.plot(np.array(data["label"][1]))
        pv.plot(np.array(data["label"][2]))
        return data

class RemapLabels(MapTransform):
    def __init__(self, keys, map_dict):
        super().__init__(keys)
        self.map_dict = map_dict

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.map_dict.get(d[key], d[key])
        return d

class ConvertToMultiChannelForBoneClassesd(MapTransform):

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 0)
            
            # Cortical class
            result.append(d[key] == 1)
            
            # Trabecular class
            result.append(d[key] == 2)
            # Stack the results
            multi_channel_label = np.stack(result, axis=0).astype(np.float32)
            ### verification ####

            # pv.plot(np.array(multi_channel_label[0]))
            # pv.plot(np.array(multi_channel_label[1]))
            # pv.plot(np.array(multi_channel_label[2]))
            d[key] = multi_channel_label
            
        return d
        
def get_train_dataloader():

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            #PrintShape(),
            ConvertToMultiChannelForBoneClassesd(keys = ['label']),
            AddChanneld(keys = ["image"]),
            #PrintShape(),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(320,320,32),
                pos = 1,neg=0),             
                
            RandFlipd(keys = ["image", "label"],
                      prob = 0.5,
                      spatial_axis = 0),
             RandFlipd(keys = ["image", "label"],
                      prob = 0.5,
                      spatial_axis = 1),
             #Affined( keys = ["image","label"],
             #       rotate_params = np.pi/4, 
             #       #shear_params= (0.2, 0.1),
             #       translate_params= (50,50),
             #       ),
            RandAdjustContrastd(
                keys = ["image"],
                gamma = (0.5, 4.5),
            ),
            RandScaleIntensityd(keys = "image", prob = 1, factors = 0.1),
            NormalizeIntensityd(keys = "image",
                                nonzero = True,
                                channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    train_ds = Dataset(data=train_files, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers = 4, pin_memory = True, )


    return train_loader

def get_val_dataloader():
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelForBoneClassesd(keys = ['label']),
            AddChanneld(keys = ["image"]),
            #DataViewImage(),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(320,320,32),
                pos = 1,neg=0),
            #DataViewImage(),
            #DataViewLabel(),
            NormalizeIntensityd(keys = "image",
                               nonzero = True,
                               channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_ds = Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers = 4, drop_last = True, pin_memory = True,)

    return val_loader

def get_test_dataloader():
    test_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelForBoneClassesd(keys = ['label']),
            AddChanneld(keys = ["image"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size = (320,320,32),
                pos = 1,neg=0),
            NormalizeIntensityd(keys = "image",
                               nonzero = True,
                               channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    test_ds = Dataset(data=test_files, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return test_loader


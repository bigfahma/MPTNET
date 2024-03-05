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
    RandAdjustContrastd,
    SpatialPadd
)   
from monai.data import DataLoader, Dataset
import json
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from torch.utils.data.dataloader import default_collate


set_determinism(seed=1)
with open('synapse_data/data_synapse.json') as f:
    data = json.load(f)
train_files, val_files, test_files = data['train'], data['val'], data['test']


class ConvertToMultiChannelForSYNAPSEClassesd(MapTransform):

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            for i in range(9):
                result.append(d[key] == i)


            multi_channel_label = np.stack(result, axis=0).astype(np.float32)
            d[key] = multi_channel_label
            
        return d
        
def get_train_dataloader():

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelForSYNAPSEClassesd(keys = ['label']),
            AddChanneld(keys = ["image"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(32, 384, 384), # MIN (Z,X,Y) : 85 500 500
                pos = 0.9, neg = 0.1),    
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
            RandScaleIntensityd(keys = "image", prob = 1, factors = 0.1),
            NormalizeIntensityd(keys = "image",
                                nonzero = True,
                                channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    train_ds = Dataset(data=train_files, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers = 4, 
                              pin_memory = True,  )


    return train_loader



def get_val_dataloader():
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelForSYNAPSEClassesd(keys = ['label']),
            AddChanneld(keys = ["image"]), 
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(32, 384, 384), # MIN (Z,X,Y) : 85 500 500
                pos = 0.9),  
            NormalizeIntensityd(keys = "image",
                               nonzero = True,
                               channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_ds = Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers = 4, 
                            pin_memory = True)

    return val_loader

def get_test_dataloader():
    test_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelForSYNAPSEClassesd(keys = ['label']),
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

def get_combined_train_val_dataloader():
    combined_files = train_files + val_files

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelForSYNAPSEClassesd(keys = ['label']),
            AddChanneld(keys = ["image"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(32, 384, 384), # MIN (Z,X,Y) : 85 500 500
                pos = 0.9),    
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
            RandScaleIntensityd(keys = "image", prob = 1, factors = 0.1),
            NormalizeIntensityd(keys = "image",
                                nonzero = True,
                                channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )    
    combined_transform = train_transform  
    combined_ds = Dataset(data=combined_files, transform=combined_transform)
    combined_loader = DataLoader(combined_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    return combined_loader


def explore_data_loader(data_loader, loader_name):
    print(f"Exploring data from {loader_name}...")

    for batch_idx, batch in enumerate(data_loader):
        print(f"\nBatch {batch_idx + 1}:")

        images = batch['image']
        labels = batch['label']

        for i in range(len(images)):
            image = images[i]
            label = labels[i]

            np_image = image.cpu().numpy()
            np_label = label.cpu().numpy()

            # Print shapes and unique values in label
            print(f"  Image {i} shape: {np_image.shape}")
            print(f"  Label {i} shape: {np_label.shape}")
            print(f"  Unique values in label {i}: {np.unique(np_label)}")
            plot = True

            if plot :
                slice_idx = np_image.shape[-1] // 2
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 10, 1)
                plt.imshow(np_image[0, :, :, slice_idx], cmap='gray')
                plt.title(f"Image {i} Slice")
                for j in range(np_label.shape[0]):
                    plt.subplot(1, 10, j + 2)
                    plt.imshow(np_label[j, :, :, slice_idx], cmap='gray')
                    plt.title(f"Label {i} Class {j} Slice")
                plt.show()

def explore_all_data():
    train_loader = get_train_dataloader()
    val_loader = get_val_dataloader()
    test_loader = get_test_dataloader()

    explore_data_loader(train_loader, "Training Data")
    explore_data_loader(val_loader, "Validation Data")
    explore_data_loader(test_loader, "Test Data")

if __name__ == "__main__":
    explore_all_data()

import pytorch_lightning as pl
from monai.data import DataLoader, Dataset
import os
import torch
import argparse
from monai.networks.nets import UNet, AttentionUnet, UNETR, SwinUNETR
from bone_data.bone_data import ConvertToMultiChannelForBoneClassesd
from pytorch_lightning.loggers import TensorBoardLogger
#from models.DBAHNet import DBAHNet
import os
from monai.utils import set_determinism
import json
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
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
if __name__ == "__main__":
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
        ])
        test_ds = Dataset(data=test_files, transform=test_transform)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        return test_loader

    print("Testing ...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(["UNET","ATTENTIONUNET","UNETR","SWINUNETR", "DBAHNET"]),default = "DBAHNET")
    parser.add_argument('--ckpt', type=str, required=True, default = "ckpt/EXP_BONE_DBAHNet/last.ckpt", help = "Insert CKPT")
    args = parser.parse_args()
    if args.model == "UNET":
         from trainer_bone_unet import BONEUNET
         model = BONEUNET().load_from_checkpoint(args.ckpt).eval()
    if args.model == "ATTENTIONUNET":
         from trainer_bone_attentionunet import BONEATTENTIONUNET
         model = BONEATTENTIONUNET().load_from_checkpoint(args.ckpt).eval()
    if args.model == "UNETR":
         from trainer_bone_unetr import BONEUNETR
         model = BONEUNETR().load_from_checkpoint(args.ckpt).eval()
    if args.model == "SWINUNETR":
         from trainer_bone_swinunetr import BONESWINUNETR
         model = BONESWINUNETR().load_from_checkpoint(args.ckpt).eval()
    if args.model == "DBAHNET":
         from trainer_bone_dbahnet import DBAHNET
         model = DBAHNET().load_from_checkpoint(args.ckpt).eval()
    set_determinism(seed=1)
    with open('bone_data/data_test_75_split.json') as f:
        data = json.load(f)
    test_files = data['test']
    test_dataloader = get_test_dataloader()
    tensorboardlogger = TensorBoardLogger(
        'logs_test',
        name = f"EXP_BONE_REGION100_{args.model}",
        default_hp_metric = None
    )
    trainer = pl.Trainer(accelerator= "gpu", precision=16, logger = tensorboardlogger)

    #print("Validation set")
    #trainer.test(model, dataloaders = val_dataloader)
    print("Testing set")
    trainer.test(model, dataloaders = test_dataloader)


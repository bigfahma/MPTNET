from acdc_data.acdc_data import get_test_dataloader
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import os


if __name__ == "__main__":

    print("Testing ...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(["UNET","ATTENTIONUNET","UNETR","SWINUNETR", "MPTNET"]),default="MPTNET", help="Choose a model from the list: [UNET, ATTENTIONUNET, UNETR, SWINUNETR, MPTNET]")

    parser.add_argument('--ckpt', type=str, required=True, default = "ckpt/EXP_BONE_MPTNET/last.ckpt", help = "Insert path to ckpt of the chosen model")
    args = parser.parse_args()
    #model = BONEUNET.load_from_checkpoint(args.ckpt).eval()
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
    if args.model == "MPTNET":
         from trainer_acdc_mptnet import MPTNET_ACDC
         model = MPTNET_ACDC().load_from_checkpoint(args.ckpt).eval()
    test_dataloader = get_test_dataloader()
    tensorboardlogger = TensorBoardLogger(
        'logs_test',
        name = f"EXP_BONE_TEST_FINAL_{args.model}",
        default_hp_metric = None
    )
    trainer = pl.Trainer(accelerator= "gpu", precision=16, logger = tensorboardlogger)

    print("Testing set")
    print("Model : ", args.model)
    trainer.test(model, dataloaders = test_dataloader)





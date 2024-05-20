from acdc_data.acdc_data import get_test_dataloader
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import os
from trainer_acdc_mptnet import MPTNET_ACDC


if __name__ == "__main__":

    print("Testing ...")
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', type=str, required=True, default = "ckpt/EXP_ACDC_MPTNET/last.ckpt", help = "Insert path to ckpt of the chosen model")
    args = parser.parse_args()

    model = MPTNET_ACDC().load_from_checkpoint(args.ckpt).eval()
    test_dataloader = get_test_dataloader()
    tensorboardlogger = TensorBoardLogger(
        'logs_test',
        name = f"EXP_ACDC_TEST_FINAL_{args.model}",
        default_hp_metric = None
    )
    trainer = pl.Trainer(accelerator= "gpu", devices = 1, precision=16, logger = tensorboardlogger)

    print("Testing set")
    print("Model : ", args.model)
    trainer.test(model, dataloaders = test_dataloader)





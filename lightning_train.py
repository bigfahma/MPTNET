from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import torch


if __name__ =="__main__":

    
    print("Number of GPUs available : ",torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=list(["ACDC","SYNAPSE", "UNET_ACDC"]),default="SYNAPSE", help="Choose a database from the list: [ACDC, SYNAPSE, UNET_ACDC]") 
    args = parser.parse_args()

    if args.dataset == "SYNAPSE":
        from trainer_synapse_mptnet import MPTNET_SYNAPSE
        model = MPTNET_SYNAPSE()
    elif args.dataset == "ACDC":
        from trainer_acdc_mptnet import MPTNET_ACDC
        model = MPTNET_ACDC()
    elif args.dataset == "UNET_ACDC":
        from trainer_acdc_unet import ACDCUNET
        model = ACDCUNET()

        
    print("Training ...")

    checkpoint_callback = ModelCheckpoint(
        monitor='val/MeanDiceScore',
        dirpath='./ckpt/{}'.format(args.exp),
        filename='Epoch{epoch:3d}-MeanDiceScore{val/MeanDiceScore:.4f}',
        save_top_k=3,
        mode='max',
        save_last= True,
        auto_insert_metric_name=False
    )
    early_stop_callback = EarlyStopping(
    monitor='val/val_loss',
    min_delta=0.0001,
    patience=50,
    verbose=True,
    mode='max'
    )
    tensorboardlogger = TensorBoardLogger(
        'logs',
        name = args.exp,
        default_hp_metric = None
    )
    trainer = pl.Trainer(
                          accelerator='gpu',
                          devices= "auto",
                            precision=16,
                            min_epochs=2000,
                            max_epochs = 3500,
                            callbacks=[ checkpoint_callback, early_stop_callback ],
                            num_sanity_val_steps=4,
                            logger = tensorboardlogger,
                            accumulate_grad_batches= 4,
                            )
    trainer.fit(model)
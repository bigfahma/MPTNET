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
    parser.add_argument('--dataset', type=str, choices=list(["ACDC","SYNAPSE", "BONE"]),default="SYNAPSE", help="Choose a database from the list: [ACDC, SYNAPSE, BONE]") 
    args = parser.parse_args()

    if args.dataset == "SYNAPSE":
        from trainer_synapse_mptnet import MPTNET_SYNAPSE
        model = MPTNET_SYNAPSE()
    elif args.dataset == "ACDC":
        from trainer_acdc_mptnet import MPTNET_ACDC
        model = MPTNET_ACDC()
    elif args.dataset == "BONE":
        from trainer_bone_mptnet import MPTNET_BONE
        model = MPTNET_BONE()

        
    print("Training ...")
    ckpt_dir = f'ckpt/{args.exp}/'
    log_dir = f'logs/{args.exp}/'
    if not os.path.exists('ckpt/{}/'.format(args.exp)):
        os.makedirs('ckpt/{}/'.format(args.exp), exist_ok=True)
        print("Created directory:", 'ckpt/{}/'.format(args.exp))
    else:
        print("Directory already exists:", 'ckpt/{}/'.format(args.exp))

    if not os.path.exists('logs/{}/'.format(args.exp)):
        os.makedirs('logs/{}/'.format(args.exp), exist_ok=True)
        print("Created directory:", 'logs/{}/'.format(args.exp))
    else:
        print("Directory already exists:", 'logs/{}/'.format(args.exp))

    checkpoint_callback = ModelCheckpoint(
        monitor='val/MeanDiceScore',
        dirpath=ckpt_dir,
        filename='Epoch{epoch:3d}-MeanDiceScore{val/MeanDiceScore:.4f}',
        save_top_k=3,
        mode='max',
        save_last= True,
        auto_insert_metric_name=False
    )
    early_stop_callback = EarlyStopping(
    monitor='val/MeanDiceScore',
    min_delta=0.0001, # 0.001, 1e-5
    patience=150, # 100, 50, 25
    verbose=True,
    mode='max'  
    )
    tensorboardlogger = TensorBoardLogger(
        save_dir=log_dir,
        name='',
        default_hp_metric=None
    )
    trainer = pl.Trainer(
                          accelerator='gpu',
                           devices = [4,5,6,7],
                            precision=16, #mixed
                            min_epochs=700,
                            max_epochs = 1500, 
                            callbacks=[ checkpoint_callback, early_stop_callback],
                            num_sanity_val_steps=2,
                            logger = tensorboardlogger,
                            check_val_every_n_epoch = 1,
                            accumulate_grad_batches= 2, # 1
                            )
    trainer.fit(model)
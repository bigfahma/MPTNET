
import torch
from monai.metrics import DiceMetric#, compute_meandice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.losses import DiceLoss,DiceFocalLoss,FocalLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activations, Compose, EnsureType
from models.MPTNet import MPTNet
import pytorch_lightning as pl
import numpy as np
from synapse_data.synapse_data import get_train_dataloader, get_val_dataloader, get_test_dataloader, get_combined_train_val_dataloader
from monai.data import decollate_batch
import matplotlib.pyplot as plt

import csv
import os

class MPTNET_SYNAPSE(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()      
        self.lr = lr
        IMAGE_DIM = (32, 384, 384)  
        NUM_CLASSES = 9
        EMB_DIM = 96
        NUM_HEADS = [6,12,24,48]
        DEPTH = 2
        WINDOW_SIZE = (4,7,7)
        NUM_LAYERS = 3
        POOLING_SIZE = [(1,2,2), (2,4,4), (2,8,8)]
        
        self.model = MPTNet(emb_dim=EMB_DIM, resolution=IMAGE_DIM, num_classes= NUM_CLASSES, depth = DEPTH, 
                   num_heads=NUM_HEADS, window_size=WINDOW_SIZE, n_layers= NUM_LAYERS, pooling_size= POOLING_SIZE,
                   emb_ratio= [1,4,4])
               
        
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch",ignore_empty=True)
        self.custom_loss = DiceCELoss(to_onehot_y=False, softmax=True, squared_pred=True)
        self.post_trans_images = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
        self.init_metrics()
        self.post_labels = Compose([EnsureType()])

    def init_metrics(self):
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.val_losses = []

    def forward(self, x):
        return self.model(x) 

    def training_step(self, batch, batch_idx):
        inputs, labels = batch['image'], batch['label']
        outputs = sliding_window_inference(inputs, (32, 384, 384), 1, self.model, overlap=0.5)
        loss = self.custom_loss(outputs, labels)
        self.log('train/loss', loss)
        return loss
    
    
    def validation_step(self, batch, batch_index):
        inputs, labels = batch['image'], batch['label']
        outputs = sliding_window_inference(inputs, (32, 384, 384), 1, self.model, overlap=0.5)
        loss = self.custom_loss(outputs, labels)
        outputs = [self.post_trans_images(val_pred_tensor) for val_pred_tensor in decollate_batch(outputs)]
        labels = [self.post_labels(val_pred_tensor)  for val_pred_tensor in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        self.val_losses.append(loss)
        return loss

    def on_validation_epoch_end(self):
        mean_val_loss = torch.stack(self.val_losses).mean()
        dice_scores = self.dice_metric.aggregate()
 
        mean_dice_c1 = dice_scores[0].item()
        mean_dice_c2 = dice_scores[1].item()
        mean_dice_c3 = dice_scores[2].item()
        mean_dice_c4 = dice_scores[3].item()
        mean_dice_c5 = dice_scores[4].item()
        mean_dice_c6 = dice_scores[5].item()
        mean_dice_c7 = dice_scores[6].item()
        mean_dice_c8 = dice_scores[7].item()

        mean_val_dice = (mean_dice_c1 + mean_dice_c2 + mean_dice_c3 + mean_dice_c4 + mean_dice_c5 + mean_dice_c6 + mean_dice_c7 + mean_dice_c8 ) / 8
        self.log('val/val_loss', mean_val_loss, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/MeanDiceScore', mean_val_dice, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class1', mean_dice_c1, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class2', mean_dice_c2, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class3', mean_dice_c3, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class4', mean_dice_c4, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class5', mean_dice_c5, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class6', mean_dice_c6, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class7', mean_dice_c7, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class8', mean_dice_c8, sync_dist= True, on_epoch=True, prog_bar=True)
        self.dice_metric.reset()
        self.val_losses.clear() 
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }        
        
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        return {"log": tensorboard_logs}    


    def test_step(self, batch, batch_index):
        inputs, labels = batch['image'], batch['label']
        outputs = sliding_window_inference(inputs,(32, 384, 384), 1, self.model, overlap=0.5)
        loss = self.custom_loss(outputs, labels)
        outputs = [self.post_trans_images(val_pred_tensor) for val_pred_tensor in decollate_batch(outputs)]
        labels = [self.post_labels(val_pred_tensor)  for val_pred_tensor in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return loss


    def on_test_epoch_end(self):
        dice_scores = self.dice_metric.aggregate()
 
        mean_dice_c1 = dice_scores[0].item()
        mean_dice_c2 = dice_scores[1].item()
        mean_dice_c3 = dice_scores[2].item()
        mean_dice_c4 = dice_scores[3].item()
        mean_dice_c5 = dice_scores[4].item()
        mean_dice_c6 = dice_scores[5].item()
        mean_dice_c7 = dice_scores[6].item()
        mean_dice_c8 = dice_scores[7].item()

        mean_test_dice = (mean_dice_c1 + mean_dice_c2 + mean_dice_c3 + mean_dice_c4 + mean_dice_c5 + mean_dice_c6 + mean_dice_c7 + mean_dice_c8 ) / 8
        print(f'Mean Test Dice score : {mean_test_dice}. \n Test dice score per class : C1 : {mean_dice_c1}, C2: {mean_dice_c2}, C3: {mean_dice_c3}, C4 : {mean_dice_c4}, C5: {mean_dice_c5}, C6: {mean_dice_c6}, C7 : {mean_dice_c7}, C8: {mean_dice_c8}')
     
        self.dice_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                    self.model.parameters(), self.lr, weight_decay=1e-5, amsgrad=True
                    )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return get_train_dataloader()   
    def val_dataloader(self):
        return get_val_dataloader()
    
    def test_dataloader(self):
        return get_test_dataloader()
    
   



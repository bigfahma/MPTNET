import torch
import torch.nn.functional as F
import torch.nn as nn 
from monai.networks.nets import UNet
from monai.metrics import DiceMetric#, compute_meandice, 
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.losses import DiceLoss,DiceFocalLoss,FocalLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activations, Compose, EnsureType
from loss.diceloss import DiceScore
import pytorch_lightning as pl
from acdc_data.acdc_data import get_train_dataloader, get_val_dataloader, get_test_dataloader

import matplotlib.pyplot as plt
import numpy as np
import csv
import os


class ACDCUNET(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.IMAGE_DIM = (128, 128, 8)
        self.NUM_CHANNELS = 1 
        self.NUM_CLASSES = 4
        self.LOSSES = {
            "Dice Loss": DiceLoss(to_onehot_y=False, sigmoid=False, squared_pred=True),
            "Dice Focal Loss": DiceFocalLoss(to_onehot_y=False, sigmoid=False, squared_pred=True),
            "Focal Loss": FocalLoss(to_onehot_y=False),
            "Dice CE Loss": DiceCELoss(to_onehot_y=False, sigmoid = True, squared_pred=True)
        }
        
        self.model = UNet(spatial_dims = 3,
                          in_channels= self.NUM_CHANNELS , 
                          out_channels= self.NUM_CLASSES,
                        channels=(32, 64, 128, 256),
                        strides=(2, 2, 2, 2))
        self.custom_loss = self.LOSSES["Dice CE Loss"]
        self.training_step_outputs = []
        self.val_losses = []
        self.test_losses = []

        self.val_mean_dice = []
        self.val_dice_c1 = []
        self.val_dice_c2 = []
        self.val_dice_c3  = []
        self.val_dice_c4 = []

        self.test_mean_dice = []
        self.test_dice_c1 = []
        self.test_dice_c2 = []
        self.test_dice_c3 = []
        self.test_dice_c4 = []

        self.test_hd_distance_c1 = []
        self.test_hd_distance_c2 = []
        self.test_hd_distance_c3 = []
        self.test_hd_distance_c4 = []
        
        self.test_hd_distance_mean = []
        self.post_trans_images = Compose(
            [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold = 0.5)]
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0

    def forward(self, x):
        return self.model(x) 
    def training_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        outputs = self.forward(inputs)
        loss = self.custom_loss(outputs, labels)
        self.training_step_outputs.append(loss)
        self.log('train/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        roi_size =  (128, 128, 8) 
        sw_batch_size = 1
        outputs = sliding_window_inference(
                inputs, roi_size, sw_batch_size, self.model, overlap = 0.5)
        loss = self.custom_loss(outputs, labels)
        self.val_losses.append(loss)
        val_outputs = self.post_trans_images(outputs)      
        metric_c1 = DiceScore(y_pred=val_outputs[:, 0:1], y=labels[:, 0:1])
        metric_c2 = DiceScore(y_pred=val_outputs[:, 1:2], y=labels[:, 1:2]) 
        metric_c3 = DiceScore(y_pred=val_outputs[:, 2:3], y=labels[:, 2:3]) 
        metric_c4 = DiceScore(y_pred=val_outputs[:, 3:4], y=labels[:, 3:4]) 

        mean_val_dice =  (metric_c2 + metric_c3 + metric_c4)/3
        self.val_mean_dice.append(mean_val_dice)
        self.val_dice_c1.append(metric_c1)
        self.val_dice_c2.append(metric_c2)
        self.val_dice_c3.append(metric_c3)
        self.val_dice_c4.append(metric_c4)

        return loss
    
    
    def on_validation_epoch_end(self):

        mean_val_loss = torch.stack(self.val_losses).mean()
        mean_val_dice = torch.stack(self.val_mean_dice).mean()
        mean_dice_c1 = torch.stack(self.val_dice_c1).mean()
        mean_dice_c2 = torch.stack(self.val_dice_c2).mean()
        mean_dice_c3 = torch.stack(self.val_dice_c3).mean()
        mean_dice_c4 = torch.stack(self.val_dice_c4).mean()

        self.log('val/val_loss', mean_val_loss, sync_dist= True, on_epoch=True)
        self.log('val/MeanDiceScore', mean_val_dice, sync_dist= True, on_epoch=True)
        self.log('val/Dice_Class1', mean_dice_c1, sync_dist= True)
        self.log('val/Dice_Class2', mean_dice_c2, sync_dist= True)
        self.log('val/Dice_Class3', mean_dice_c3, sync_dist= True)
        self.log('val/Dice_Class4', mean_dice_c4, sync_dist= True) 

        os.makedirs(self.logger.log_dir,  exist_ok=True)
        self.val_losses.clear() 
        self.val_mean_dice.clear() 
        self.val_dice_c1.clear()  
        self.val_dice_c2.clear()  
        self.val_dice_c3.clear()  
        self.val_dice_c4.clear()  

        if self.current_epoch == 0:
            with open('{}/metric_log.csv'.format(self.logger.log_dir), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', "ValLoss",'Mean Dice Score', 'DiceClass1', 'DiceClass2',"DiceClass3", "DiceClass4"])
        with open('{}/metric_log.csv'.format(self.logger.log_dir), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.current_epoch, mean_val_loss.detach(), mean_val_dice.detach(), mean_dice_c1.detach(), 
                             mean_dice_c2.detach(), mean_dice_c3.detach(), mean_dice_c4.detach()])

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
                f"\n Current epoch: {self.current_epoch} Current mean dice: {mean_val_dice:.4f}"
                f" Class1: {mean_dice_c1:.4f} Class2: {mean_dice_c2:.4f} Class3: {mean_dice_c3:.4f} Class4: {mean_dice_c4:.4f}"
                f"\n Best mean dice: {self.best_val_dice}"
                f" at epoch: {self.best_val_epoch}"
            )
        return {'val_MeanDiceScore': mean_val_dice}
    
    def test_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        roi_size = (128, 128, 8) 
        sw_batch_size = 1
        test_outputs = sliding_window_inference(
                    inputs, roi_size, sw_batch_size, self.forward, overlap = 0.5)
        loss = self.custom_loss(test_outputs, labels)
        test_outputs = self.post_trans_images(test_outputs)
        self.test_losses.append(loss)
        dice_c1 = DiceScore(y_pred=test_outputs[:, 0:1], y=labels[:, 0:1])
        dice_c2 = DiceScore(y_pred=test_outputs[:, 1:2], y=labels[:, 1:2]) 
        dice_c3 = DiceScore(y_pred=test_outputs[:, 2:3], y=labels[:, 2:3]) 
        dice_c4 = DiceScore(y_pred=test_outputs[:, 3:4], y=labels[:, 3:4])

        mean_test_dice = (dice_c2 + dice_c3 + dice_c4) / 3
        spacing = (0.005, 0.005, 0.005)  # Spacing in mm (5µm = 0.005 mm)
        hd_c1 = compute_hausdorff_distance(y_pred = test_outputs[:, 0:1], y = labels[:,0:1], include_background = False, spacing =  spacing, percentile = 95)
        hd_c2 = compute_hausdorff_distance(y_pred = test_outputs[:, 1:2], y = labels[:,1:2], include_background = False, spacing =  spacing, percentile = 95)
        hd_c3 = compute_hausdorff_distance(y_pred = test_outputs[:, 2:3], y = labels[:,2:3], include_background = False, spacing =  spacing, percentile = 95)
        hd_c4 = compute_hausdorff_distance(y_pred = test_outputs[:, 3:4], y = labels[:,3:4], include_background = False, spacing =  spacing, percentile = 95)

        hd_mean =  (hd_c2 + hd_c3 + hd_c4)/3
        self.test_hd_distance_mean.append(hd_mean)
        self.test_hd_distance_c1.append(hd_c1)
        self.test_hd_distance_c2.append(hd_c2)
        self.test_hd_distance_c3.append(hd_c3)
        self.test_hd_distance_c4.append(hd_c4)

        self.test_mean_dice.append(mean_test_dice)
        self.test_dice_c1.append(dice_c1)
        self.test_dice_c2.append(dice_c2)
        self.test_dice_c3.append(dice_c3)
        self.test_dice_c4.append(dice_c4)

        return loss

    def on_test_epoch_end(self):
        test_loss = torch.stack(self.test_losses).mean()
        mean_test_dice = torch.stack(self.test_mean_dice).mean()
        dice_c1 = torch.stack(self.test_dice_c1).mean()
        dice_c2 = torch.stack(self.test_dice_c2).mean()
        dice_c3 = torch.stack(self.test_dice_c3).mean()
        dice_c4 = torch.stack(self.test_dice_c4).mean()

        hd_c1 = torch.stack(self.test_hd_distance_c1).mean()
        hd_c2 = torch.stack(self.test_hd_distance_c2).mean()
        hd_c3 = torch.stack(self.test_hd_distance_c3).mean()
        hd_c4 = torch.stack(self.test_hd_distance_c4).mean()
        hd_mean = torch.stack(self.test_hd_distance_mean).mean()

        self.log('test/test_loss', test_loss, sync_dist= True)
        self.log('test/MeanDiceScore', mean_test_dice, sync_dist= True)
        self.log('test/DiceClass1', dice_c1, sync_dist= True)
        self.log('test/DiceClass2', dice_c2, sync_dist= True)
        self.log('test/DiceClass3', dice_c3, sync_dist= True)
        self.log('test/DiceClass4', dice_c4, sync_dist= True)
        self.log('test/Hausdorff Distance 95% Class1', hd_c1, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Class2', hd_c2, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Class3', hd_c3, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Class4', hd_c4, sync_dist = True) 

        self.test_losses.clear()  # free memory
        with open('{}/test_log.csv'.format(self.logger.log_dir), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Mean Test Dice", "TestLoss","DiceClass1", "DiceClass2", "DiceClass3", "DiceClass4", "Hausdorff Distance 95% Class1", "Hausdorff Distance 95% Class2", "Hausdorff Distance 95% Class3", "Hausdorff Distance 95% Class4", "Hausdorff Distance 95% Mean"])
            writer.writerow([mean_test_dice, test_loss, dice_c1, dice_c2, dice_c3, dice_c4, hd_c1, hd_c2, hd_c3, hd_c4, hd_mean])
        return {'test_MeanDiceScore': mean_test_dice}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                    self.model.parameters(), self.lr, weight_decay=1e-5, amsgrad=True
                    )
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 3000, 0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return get_train_dataloader()
    
    def val_dataloader(self):
        return get_val_dataloader()
    
    def test_dataloader(self):
        return get_test_dataloader()
    
   



# Torch Library
import torch
import torch.nn.functional as F
import torch.nn as nn 
# MONAI
from monai.networks.nets import UNETR
from monai.metrics import DiceMetric#, compute_meandice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.losses import DiceLoss,DiceFocalLoss,FocalLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activations, Compose, EnsureType
from loss.diceloss import DiceScore

# Pytorch Lightning
import pytorch_lightning as pl

# Custom Libraries
from bone_data.bone_data import get_train_dataloader, get_val_dataloader, get_test_dataloader

import matplotlib.pyplot as plt

import csv
import os

class BONEUNETR(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.IMAGE_DIM = (320,320, 32)    
        self.PATCH_DIM = 8
        self.NUM_CHANNELS = 1 
        self.NUM_CLASSES = 3
        self.LOSSES = {
            #"Dice Loss": DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True),
            "Dice Focal Loss": DiceFocalLoss(to_onehot_y=False, sigmoid=False, squared_pred=True),
            "Focal Loss": FocalLoss(to_onehot_y=False),
            "Dice CE Loss": DiceCELoss(to_onehot_y=False, sigmoid=False, squared_pred=True)
        }
        
        self.model = UNETR(in_channels= 1,
                           out_channels= 3,
                           img_size= self.IMAGE_DIM,
                            ####DEFAULT VALUES BELOW ###
                           feature_size= 32,
                           hidden_size = 768,
                           mlp_dim = 3072,
                           num_heads=8,
                           pos_embed="conv",
                           norm_name="instance")
        
        # self.pretrained_model = torch.load("UNETR_model_best_acc.pth")
        # self.model.load_state_dict(self.pretrained_model, strict= False)

        self.custom_loss = self.LOSSES["Dice CE Loss"]
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        ##### metrics tracking #####
        self.val_mean_dice = []
        self.val_dice_cortical = []
        self.val_dice_trabecular = []

        self.test_mean_dice = []
        self.test_dice_cortical = []
        self.test_dice_trabecular = []
        self.test_hd_distance_trab = []
        self.test_hd_distance_cor = []
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
        roi_size =  (320,320,32)
        sw_batch_size = 2
        outputs = sliding_window_inference(
                inputs, roi_size, sw_batch_size, self.model, overlap = 0.5)
        loss = self.custom_loss(outputs, labels)
        self.validation_step_outputs.append(loss)
        val_outputs = self.post_trans_images(outputs)
        metric_cor = DiceScore(y_pred=val_outputs[:, 0:1], y=labels[:, 0:1], include_background= False)
        metric_trab = DiceScore(y_pred=val_outputs[:, 1:2], y=labels[:, 1:2], include_background = False) # Change maybe include background
        mean_val_dice =  (metric_trab + metric_cor)/2
        self.val_mean_dice.append(mean_val_dice)
        self.val_dice_cortical.append(metric_cor)
        self.val_dice_trabecular.append(metric_trab)
        
        return loss
    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        mean_val_dice = torch.stack(self.val_mean_dice).mean()
        metric_cor = torch.stack(self.val_dice_cortical).mean()
        metric_trab = torch.stack(self.val_dice_trabecular).mean()
        self.log('val/Loss', epoch_average, sync_dist= True)
        self.log('val/MeanDiceScore', mean_val_dice, sync_dist= True)
        self.log('val/DiceCortical', metric_cor, sync_dist= True)
        self.log('val/DiceTrabecular', metric_trab, sync_dist= True)
        os.makedirs(self.logger.log_dir,  exist_ok=True)
        self.validation_step_outputs.clear() 
        self.val_mean_dice.clear()  
        self.val_dice_cortical.clear()  
        self.val_dice_trabecular.clear()  

        if self.current_epoch == 0:
            with open('{}/metric_log.csv'.format(self.logger.log_dir), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Mean Dice Score', 'DiceCortical', 'DiceTrabecular'])
        with open('{}/metric_log.csv'.format(self.logger.log_dir), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.current_epoch, mean_val_dice.item(), metric_cor.item(), metric_trab.item()])

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
                f"\n Current epoch: {self.current_epoch} Current mean dice: {mean_val_dice:.4f}"
                f" Cortical: {metric_cor:.4f} Trabecular: {metric_trab:.4f}"
                f"\n Best mean dice: {self.best_val_dice}"
                f" at epoch: {self.best_val_epoch}"
            )
        return {'val_MeanDiceScore': mean_val_dice}
    
    def test_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        roi_size = (320,320,32)
        sw_batch_size = 1
        test_outputs = sliding_window_inference(
                    inputs, roi_size, sw_batch_size, self.forward, overlap = 0.5)
        loss = self.custom_loss(test_outputs, labels)
        test_outputs = self.post_trans_images(test_outputs)
        self.test_step_outputs.append(loss)
        metric_cor = DiceScore(y_pred=test_outputs[:, 0:1], y=labels[:, 0:1], include_background = False)
        metric_trab = DiceScore(y_pred=test_outputs[:, 1:2], y=labels[:, 1:2], include_background = False)
        mean_test_dice =  (metric_cor + metric_trab )/2
        spacing = (0.005, 0.005, 0.005)  # Spacing in mm (5µm = 0.005 mm)
        hd_cor = compute_hausdorff_distance(y_pred = test_outputs[:, 0:1], y = labels[:,0:1], include_background = False, spacing =  spacing, percentile = 95)
        hd_trab = compute_hausdorff_distance(y_pred = test_outputs[:, 1:2], y = labels[:,1:2], include_background = False, spacing =  spacing, percentile = 95)
        hd_mean =  (hd_cor + hd_trab)/2
        self.test_hd_distance_mean.append(hd_mean)
        self.test_hd_distance_trab.append(hd_trab)
        self.test_hd_distance_cor.append(hd_cor)
        self.test_mean_dice.append(mean_test_dice)
        self.test_dice_cortical.append(metric_cor)
        self.test_dice_trabecular.append(metric_trab)
        return loss

    def on_test_epoch_end(self):
        epoch_average = torch.stack(self.test_step_outputs).mean()
        mean_test_dice = torch.stack(self.test_mean_dice).mean()
        metric_cor = torch.stack(self.test_dice_cortical).mean()
        metric_trab = torch.stack(self.test_dice_trabecular).mean()
        metric_hd_cor = torch.stack(self.test_hd_distance_cor).mean()
        metric_hd_trab = torch.stack(self.test_hd_distance_trab).mean()
        metric_hd_mean = torch.stack(self.test_hd_distance_mean).mean()
        self.log('test/Loss', epoch_average, sync_dist= True)
        self.log('test/MeanDiceScore', mean_test_dice, sync_dist= True)
        self.log('test/DiceCortical', metric_cor, sync_dist= True)
        self.log('test/DiceTrabecular', metric_trab, sync_dist= True)
        self.log('test/Hausdorff Distance 95% Cortical', metric_hd_cor, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Trabecular', metric_hd_trab, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Mean', metric_hd_mean, sync_dist = True)
        self.test_step_outputs.clear()  # free memory
        with open('{}/test_log.csv'.format(self.logger.log_dir), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Mean Test Dice", "DiceCortical", "DiceTrabecular","Hausdorff Distance 95% Cortical", "Hausdorff Distance 95% Trabecular", "Hausdorff Distance 95% Mean"])
            writer.writerow([mean_test_dice, metric_cor, metric_trab, metric_hd_cor, metric_hd_trab, metric_hd_mean])
        return {'test_MeanDiceScore': mean_test_dice}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                    self.model.parameters(), self.lr, weight_decay=1e-5, amsgrad=True
                    )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return get_train_dataloader()
    
    def val_dataloader(self):
        return get_val_dataloader()
    
    def test_dataloader(self):
        return get_test_dataloader()
    
   


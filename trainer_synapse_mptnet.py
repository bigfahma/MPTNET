
import torch
from monai.metrics import DiceMetric#, compute_meandice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.losses import DiceLoss,DiceFocalLoss,FocalLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activations, Compose, EnsureType
from loss.diceloss import DiceScore
from MPTNet import MPTNet
import pytorch_lightning as pl
import numpy as np
from synapse_data.synapse_data import get_train_dataloader, get_val_dataloader, get_test_dataloader, get_combined_train_val_dataloader
from monai.data import decollate_batch
import matplotlib.pyplot as plt

import csv
import os

class MPTNET_SYNAPSE(pl.LightningModule):
    def __init__(self, lr=1e-5):
        super().__init__()      
        self.lr = lr
        IMAGE_DIM = (32, 384,384)    
        NUM_CLASSES = 9
        EMB_DIM = 96
        NUM_HEADS = [6,12,24,48]
        DEPTH = 2
        WINDOW_SIZE = (7,7,7)
        NUM_LAYERS = 3
        POOLING_SIZE = [(1,2,2), (1,4,4), (1,8,8)]
        LOSSES = {
            "Dice CE Loss": DiceCELoss(to_onehot_y=False, softmax=True)

        }
        
        self.model = MPTNet(emb_dim=EMB_DIM, resolution=IMAGE_DIM, num_classes= NUM_CLASSES, depth = DEPTH, 
                   num_heads=NUM_HEADS, window_size=WINDOW_SIZE, n_layers= NUM_LAYERS, pooling_size= POOLING_SIZE,
                   emb_ratio= [4,4,4])
               
        self.custom_loss = LOSSES["Dice CE Loss"]


        self.training_losses = []
        self.val_losses = []
        self.test_losses = []

        ##### metrics tracking #####
        self.val_mean_dice = []
        self.val_dice_c1 = []
        self.val_dice_c2 = []
        self.val_dice_c3  = []
        self.val_dice_c4 = []
        self.val_dice_c5 = []
        self.val_dice_c6  = []
        self.val_dice_c7 = []
        self.val_dice_c8 = []
        self.val_dice_c9 = []

        self.test_mean_dice = []
        self.test_dice_c1 = []
        self.test_dice_c2 = []
        self.test_dice_c3 = []
        self.test_dice_c4 = []
        self.test_dice_c5 = []
        self.test_dice_c6 = []
        self.test_dice_c7 = []
        self.test_dice_c8 = []
        self.test_dice_c9 = []

        self.test_hd_distance_c1 = []
        self.test_hd_distance_c2 = []
        self.test_hd_distance_c3 = []
        self.test_hd_distance_c4 = []
        self.test_hd_distance_c5 = []
        self.test_hd_distance_c6 = []
        self.test_hd_distance_c7 = []
        self.test_hd_distance_c8 = []
        self.test_hd_distance_c9 = []
        self.test_hd_distance_mean = []

        self.post_trans_images = Compose(
            [EnsureType(), AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)]
        )

        self.best_val_dice = 0
        self.best_val_epoch = 0


    def forward(self, x):
        return self.model(x) 
    def training_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        outputs = self.forward(inputs)
        loss = self.custom_loss(outputs, labels)
        self.training_losses.append(loss)
        os.makedirs(self.logger.log_dir,  exist_ok=True)

        self.log('train/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        roi_size =  (32, 384,384) 
        sw_batch_size = 2
        outputs = sliding_window_inference(
                inputs, roi_size, sw_batch_size, self.model, overlap = 0.5)
        val_loss = self.custom_loss(outputs, labels)
        self.val_losses.append(val_loss)
        val_outputs = torch.stack([self.post_trans_images(i) for i in decollate_batch(outputs)])

        dice_c1 = DiceScore(y_pred=val_outputs[:, 0:1], y=labels[:, 0:1])
        dice_c2 = DiceScore(y_pred=val_outputs[:, 1:2], y=labels[:, 1:2]) 
        dice_c3 = DiceScore(y_pred=val_outputs[:, 2:3], y=labels[:, 2:3]) 
        dice_c4 = DiceScore(y_pred=val_outputs[:, 3:4], y=labels[:, 3:4])
        dice_c5 = DiceScore(y_pred=val_outputs[:, 4:5], y=labels[:, 4:5]) 
        dice_c6 = DiceScore(y_pred=val_outputs[:, 5:6], y=labels[:, 5:6]) 
        dice_c7 = DiceScore(y_pred=val_outputs[:, 6:7], y=labels[:, 6:7])
        dice_c8 = DiceScore(y_pred=val_outputs[:, 7:8], y=labels[:, 7:8]) 
        dice_c9 = DiceScore(y_pred=val_outputs[:, 8:9], y=labels[:, 8:9]) 

        mean_val_dice =  (dice_c2 + dice_c3 + dice_c4 + dice_c5 + dice_c6 + dice_c7 + dice_c8 + dice_c9)/8
        self.val_mean_dice.append(mean_val_dice)
        self.val_dice_c1.append(dice_c1)
        self.val_dice_c2.append(dice_c2)
        self.val_dice_c3.append(dice_c3)
        self.val_dice_c4.append(dice_c4)
        self.val_dice_c5.append(dice_c5)
        self.val_dice_c6.append(dice_c6)
        self.val_dice_c7.append(dice_c7)
        self.val_dice_c8.append(dice_c8)
        self.val_dice_c9.append(dice_c9)
                    
        return val_loss
    
    def on_validation_epoch_end(self):
        mean_val_loss = torch.stack(self.val_losses).mean()
        mean_val_dice = torch.stack(self.val_mean_dice).mean()
        mean_dice_c1 = torch.stack(self.val_dice_c1).mean()
        mean_dice_c2 = torch.stack(self.val_dice_c2).mean()
        mean_dice_c3 = torch.stack(self.val_dice_c3).mean()
        mean_dice_c4 = torch.stack(self.val_dice_c4).mean()
        mean_dice_c5 = torch.stack(self.val_dice_c5).mean()
        mean_dice_c6 = torch.stack(self.val_dice_c6).mean()
        mean_dice_c7 = torch.stack(self.val_dice_c7).mean()
        mean_dice_c8 = torch.stack(self.val_dice_c8).mean()
        mean_dice_c9 = torch.stack(self.val_dice_c9).mean()
       
        self.val_losses.clear() 
        self.val_mean_dice.clear() 
        self.val_dice_c1.clear()  
        self.val_dice_c2.clear()  
        self.val_dice_c3.clear()  
        self.val_dice_c4.clear()  
        self.val_dice_c5.clear()  
        self.val_dice_c6.clear()  
        self.val_dice_c7.clear()  
        self.val_dice_c8.clear()  
        self.val_dice_c9.clear()  

        self.log('val/val_loss', mean_val_loss, sync_dist= True, on_epoch=True)
        self.log('val/MeanDiceScore', mean_val_dice, sync_dist= True, on_epoch=True)
        self.log('val/Dice_Class1', mean_dice_c1, sync_dist= True)
        self.log('val/Dice_Class2', mean_dice_c2, sync_dist= True)
        self.log('val/Dice_Class3', mean_dice_c3, sync_dist= True)
        self.log('val/Dice_Class4', mean_dice_c4, sync_dist= True)
        self.log('val/Dice_Class5', mean_dice_c5, sync_dist= True)
        self.log('val/Dice_Class6', mean_dice_c6, sync_dist= True)
        self.log('val/Dice_Class7', mean_dice_c7, sync_dist= True)
        self.log('val/Dice_Class8', mean_dice_c8, sync_dist= True)
        self.log('val/Dice_Class9', mean_dice_c9, sync_dist= True)

        if self.current_epoch == 0:
            with open('{}/metric_log.csv'.format(self.logger.log_dir), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', "Mean Val Loss",'Mean Dice Score', 'DiceClass1', 'DiceClass2',"DiceClass3",'DiceClass4', 'DiceClass5',"DiceClass6",'DiceClass7', 'DiceClass8', 'DiceClass9'])
        with open('{}/metric_log.csv'.format(self.logger.log_dir), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.current_epoch, mean_val_loss.detach(), mean_val_dice.detach(), mean_dice_c1.detach(), mean_dice_c2.detach(), mean_dice_c3.detach(),
                             mean_dice_c4.detach(), mean_dice_c5.detach(), mean_dice_c6.detach(),mean_dice_c7.detach(), mean_dice_c8.detach(), mean_dice_c9.detach()])

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
                f"\n Current epoch: {self.current_epoch} Current mean dice: {mean_val_dice:.4f}"
                f" Class1: {mean_dice_c1:.4f} Class2: {mean_dice_c2:.4f} Class3: {mean_dice_c3:.4f} Class4: {mean_dice_c4:.4f} Class5: {mean_dice_c5:.4f} Class6: {mean_dice_c6:.4f} Class7: {mean_dice_c7:.4f} Class8: {mean_dice_c8:.4f} Class9: {mean_dice_c9:.4f}"
                f"\n Best mean dice: {self.best_val_dice}"
                f" at epoch: {self.best_val_epoch}"
            )
        return {'val_MeanDiceScore': mean_val_dice}
    
    def test_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        roi_size = (32, 384,384)
        sw_batch_size = 1
        test_outputs = sliding_window_inference(
                    inputs, roi_size, sw_batch_size, self.forward, overlap = 0.5)
        test_loss = self.custom_loss(test_outputs, labels)
        test_outputs = torch.stack([self.post_trans_images(i) for i in decollate_batch(test_outputs)])
        self.test_losses.append(test_loss)
        dice_c1 = DiceScore(y_pred=test_outputs[:, 0:1], y=labels[:, 0:1])
        dice_c2 = DiceScore(y_pred=test_outputs[:, 1:2], y=labels[:, 1:2]) 
        dice_c3 = DiceScore(y_pred=test_outputs[:, 2:3], y=labels[:, 2:3]) 
        dice_c4 = DiceScore(y_pred=test_outputs[:, 3:4], y=labels[:, 3:4])
        dice_c5 = DiceScore(y_pred=test_outputs[:, 4:5], y=labels[:, 4:5]) 
        dice_c6 = DiceScore(y_pred=test_outputs[:, 5:6], y=labels[:, 5:6])
        dice_c7 = DiceScore(y_pred=test_outputs[:, 6:7], y=labels[:, 6:7])
        dice_c8 = DiceScore(y_pred=test_outputs[:, 7:8], y=labels[:, 7:8])
        dice_c9 = DiceScore(y_pred=test_outputs[:, 8:9], y=labels[:, 8:9])

        mean_test_dice = (dice_c2 + dice_c3 + dice_c4 + dice_c5 + dice_c6 + dice_c7 + dice_c8 + dice_c9) / 8
        spacing = (0.005, 0.005, 0.005)  # Spacing in mm (5µm = 0.005 mm)
        hd_c1 = compute_hausdorff_distance(y_pred = test_outputs[:, 0:1], y = labels[:,0:1], spacing =  spacing, percentile = 95)
        hd_c2 = compute_hausdorff_distance(y_pred = test_outputs[:, 1:2], y = labels[:,1:2], spacing =  spacing, percentile = 95)
        hd_c3 = compute_hausdorff_distance(y_pred = test_outputs[:, 2:3], y = labels[:,2:3], spacing =  spacing, percentile = 95)
        hd_c4 = compute_hausdorff_distance(y_pred = test_outputs[:, 3:4], y = labels[:,3:4], spacing =  spacing, percentile = 95)
        hd_c5 = compute_hausdorff_distance(y_pred = test_outputs[:, 4:5], y = labels[:,4:5], spacing =  spacing, percentile = 95)
        hd_c6 = compute_hausdorff_distance(y_pred = test_outputs[:, 5:6], y = labels[:,5:6], spacing =  spacing, percentile = 95)
        hd_c7 = compute_hausdorff_distance(y_pred = test_outputs[:, 6:7], y = labels[:,6:7], spacing =  spacing, percentile = 95)
        hd_c8 = compute_hausdorff_distance(y_pred = test_outputs[:, 7:8], y = labels[:,7:8], spacing =  spacing, percentile = 95)
        hd_c9 = compute_hausdorff_distance(y_pred = test_outputs[:, 8:9], y = labels[:,8:9], spacing =  spacing, percentile = 95)

        hd_mean =  ( hd_c2 + hd_c3 + hd_c4 + hd_c5 + hd_c6 + hd_c7 + hd_c8 + hd_c9 )/ 8
        self.test_hd_distance_mean.append(hd_mean)
        self.test_hd_distance_c1.append(hd_c1)
        self.test_hd_distance_c2.append(hd_c2)
        self.test_hd_distance_c3.append(hd_c3)
        self.test_hd_distance_c4.append(hd_c4)
        self.test_hd_distance_c5.append(hd_c5)
        self.test_hd_distance_c6.append(hd_c6)
        self.test_hd_distance_c7.append(hd_c7)
        self.test_hd_distance_c8.append(hd_c8)
        self.test_hd_distance_c9.append(hd_c9)

        self.test_mean_dice.append(mean_test_dice)
        self.test_dice_c1.append(dice_c1)
        self.test_dice_c2.append(dice_c2)
        self.test_dice_c3.append(dice_c3)
        self.test_dice_c4.append(dice_c4)
        self.test_dice_c5.append(dice_c5)
        self.test_dice_c6.append(dice_c6)
        self.test_dice_c7.append(dice_c7)
        self.test_dice_c8.append(dice_c8)
        self.test_dice_c8.append(dice_c9)

        self.log('test/test_loss', test_loss, sync_dist= True)
        self.log('test/MeanDiceScore', mean_test_dice, sync_dist= True)
        self.log('test/DiceClass1', dice_c1, sync_dist= True)
        self.log('test/DiceClass2', dice_c2, sync_dist= True)
        self.log('test/DiceClass3', dice_c3, sync_dist= True)
        self.log('test/DiceClass4', dice_c4, sync_dist= True)
        self.log('test/DiceClass5', dice_c5, sync_dist= True)
        self.log('test/DiceClass6', dice_c6, sync_dist= True)
        self.log('test/DiceClass7', dice_c7, sync_dist= True)
        self.log('test/DiceClass8', dice_c8, sync_dist= True)
        self.log('test/DiceClass9', dice_c9, sync_dist= True)
        
        self.log('test/Hausdorff Distance 95% Class1', hd_c1, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Class2', hd_c2, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Class3', hd_c3, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Class4', hd_c4, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Class5', hd_c5, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Class6', hd_c6, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Class7', hd_c7, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Class8', hd_c8, sync_dist = True)
        self.log('test/Hausdorff Distance 95% Class9', hd_c9, sync_dist = True)

        self.log('test/Hausdorff Distance 95% Mean', hd_mean, sync_dist = True)
        
        return test_loss

    def on_test_epoch_end(self):
        test_loss = torch.stack(self.test_losses).mean()
        mean_test_dice = torch.stack(self.test_mean_dice).mean()
        dice_c1 = torch.stack(self.test_dice_c1).mean()
        dice_c2 = torch.stack(self.test_dice_c2).mean()
        dice_c3 = torch.stack(self.test_dice_c3).mean()
        dice_c4 = torch.stack(self.test_dice_c4).mean()
        dice_c5 = torch.stack(self.test_dice_c5).mean()
        dice_c6 = torch.stack(self.test_dice_c6).mean()
        dice_c7 = torch.stack(self.test_dice_c7).mean()
        dice_c8 = torch.stack(self.test_dice_c8).mean()
        dice_c9 = torch.stack(self.test_dice_c9).mean()
        
        hd_c1 = torch.stack(self.test_hd_distance_c1).mean()
        hd_c2 = torch.stack(self.test_hd_distance_c2).mean()
        hd_c3 = torch.stack(self.test_hd_distance_c3).mean()
        hd_c4 = torch.stack(self.test_hd_distance_c4).mean()
        hd_c5 = torch.stack(self.test_hd_distance_c5).mean()
        hd_c6 = torch.stack(self.test_hd_distance_c6).mean()
        hd_c7 = torch.stack(self.test_hd_distance_c7).mean()
        hd_c8 = torch.stack(self.test_hd_distance_c8).mean()
        hd_c9 = torch.stack(self.test_hd_distance_c9).mean()
        
        hd_mean = torch.stack(self.test_hd_distance_mean).mean()

        self.test_losses.clear() 
        with open('{}/test_log.csv'.format(self.logger.log_dir), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Mean Test Loss","Mean Test Dice", "DiceClass1", "DiceClass2", "DiceClass3","DiceClass4", "DiceClass5", 
                             "DiceClass6","DiceClass7", "DiceClass8","DiceClass9", "Hausdorff Distance 95% Class1", 
                             "Hausdorff Distance 95% Class2", "Hausdorff Distance 95% Class3", "Hausdorff Distance 95% Class4", 
                             "Hausdorff Distance 95% Class5","Hausdorff Distance 95% Class6", "Hausdorff Distance 95% Class7", 
                             "Hausdorff Distance 95% Class8", "Hausdorff Distance 95% Class9", "Hausdorff Distance 95% Mean"])
            writer.writerow([test_loss, mean_test_dice, dice_c1, dice_c2, dice_c3, dice_c4, dice_c5, dice_c6, dice_c7, dice_c8, dice_c9,
                             hd_c1, hd_c2, hd_c3,hd_c4, hd_c5, hd_c6,
                             hd_c7, hd_c8, hd_c9, hd_mean])
        return {'test_MeanDiceScore': mean_test_dice}
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), self.lr, momentum = 0.9, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(
        #             self.model.parameters(), self.lr, weight_decay=1e-4, amsgrad=True
        #             )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
        return [optimizer], [scheduler]
    
    # def train_dataloader(self):
    #     return get_combined_train_val_dataloader()

    def train_dataloader(self):
        return get_train_dataloader()   
    def val_dataloader(self):
        return get_val_dataloader()
    
    def test_dataloader(self):
        return get_test_dataloader()
    
   



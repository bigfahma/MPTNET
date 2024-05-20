
import torch
from monai.metrics import DiceMetric
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.losses import DiceLoss,DiceFocalLoss,FocalLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activations, Compose, EnsureType
from models.MPTNet import MPTNet
import pytorch_lightning as pl
from synapse_data.synapse_data import get_train_dataloader, get_val_dataloader, get_test_dataloader, get_combined_train_val_dataloader
from monai.data import decollate_batch

class MPTNET_BONE(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()      
        self.lr = lr
        IMAGE_DIM = (32, 320, 320)  
        NUM_CLASSES = 3
        EMB_DIM = 96
        NUM_HEADS = [6,12,24,48]
        DEPTH = 2
        WINDOW_SIZE = (4,7,7)
        NUM_LAYERS = 3
        POOLING_SIZE = [(1,2,2), (2,4,4), (2,8,8)]
        
        self.model = MPTNet(emb_dim=EMB_DIM, resolution=IMAGE_DIM, num_classes= NUM_CLASSES, depth = DEPTH, 
                   num_heads=NUM_HEADS, window_size=WINDOW_SIZE, n_layers= NUM_LAYERS, pooling_size= POOLING_SIZE,
                   emb_ratio= [2,4,4])
               
        
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
        outputs = sliding_window_inference(inputs,(32, 320, 320) , 1, self.model, overlap=0.5)
        loss = self.custom_loss(outputs, labels)
        self.log('train/loss', loss)
        return loss
    
    
    def validation_step(self, batch, batch_index):
        inputs, labels = batch['image'], batch['label']
        outputs = sliding_window_inference(inputs,(32, 320, 320)  , 1, self.model, overlap=0.5)
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
        mean_val_dice = (mean_dice_c1 + mean_dice_c2 ) / 2
        self.log('val/val_loss', mean_val_loss, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/MeanDiceScore', mean_val_dice, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class1', mean_dice_c1, sync_dist= True, on_epoch=True, prog_bar=True)
        self.log('val/Dice_Class2', mean_dice_c2, sync_dist= True, on_epoch=True, prog_bar=True)
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
        outputs = sliding_window_inference(inputs,(32, 320, 320), 1, self.model, overlap=0.5)
        loss = self.custom_loss(outputs, labels)
        outputs = [self.post_trans_images(val_pred_tensor) for val_pred_tensor in decollate_batch(outputs)]
        labels = [self.post_labels(val_pred_tensor)  for val_pred_tensor in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return loss


    def on_test_epoch_end(self):
        dice_scores = self.dice_metric.aggregate()
 
        mean_dice_c1 = dice_scores[0].item()
        mean_dice_c2 = dice_scores[1].item()
        mean_test_dice = (mean_dice_c1 + mean_dice_c2 ) / 2
        print(f'Mean Test Dice score : {mean_test_dice}. \n Test dice score per class : C1 : {mean_dice_c1}, C2: {mean_dice_c2}')
     
        self.dice_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                    self.model.parameters(), self.lr, weight_decay=1e-5, amsgrad=True
                    )
        #optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum = 0.99)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min', 
        #     factor=0.1, 
        #     patience=40,  
        #     min_lr=1e-7, 
        #     verbose=True)

        return [optimizer], [scheduler]

    def train_dataloader(self):
        return get_train_dataloader()   
    def val_dataloader(self):
        return get_val_dataloader()
    
    def test_dataloader(self):
        return get_test_dataloader()
    
   



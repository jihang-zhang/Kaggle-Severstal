import time
import random
import os
import sys
sys.path.insert(0, '../src/networks')
sys.path.insert(0, '../src/layers')
sys.path.insert(0, '../src/layers/backbone')
sys.path.insert(0, '../src/data')
sys.path.insert(0, '../src/utils')

import yaml
import h5py
import logging
import functools

import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, fbeta_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_pretrained_bert.optimization import WarmupConstantSchedule

from tqdm import tqdm

from networks.cls_seg_unet_efficientnet_lighter import EfficientUnet
from layers.seg_losses import *
from data.datasets import *
from utils.log_utils import create_folder, create_logging


# Configuration
SEED = 12536
LOG_INTERVAL = 100
MAX_EPOCHS = 100
STAGE_EPOCHS = 50
BATCH_SIZE = 8
FOLD = 0
MODEL_TYPE = 'efficientnet-b3'
SAMPLE_TYPE = 1
MASK_THR = 0
SAVE_PATH = '../output/unet_{}_lighter_cls_seg_balance_binary_crop_constant'.format(MODEL_TYPE)
create_folder(SAVE_PATH)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = '../input/severstal-steel-defect-detection/train_images'
LOGS_DIR = os.path.join(SAVE_PATH, 'logs')

# Training Hyperparameters
warmup_proportion = 0.005
lr = 0.00015
accumulation_steps = 1


# Set the random seed manually for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

random.seed(SEED)
np.random.seed(SEED)


# Load folds
hf = h5py.File('../input/severstal-steel-defect-detection/cv_folds_10.h5', 'r')
train_ids = np.array(hf['fold{}_train'.format(FOLD)]).astype(str)
valid_ids = np.array(hf['fold{}_valid'.format(FOLD)]).astype(str)
hf.close()


# Load training df
full_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
full_df['ImageId'] = full_df['ImageId_ClassId'].apply(lambda x: x[:-2])
full_df['ClassId'] = full_df['ImageId_ClassId'].apply(lambda x: x[-1]).astype(int)
full_df['target'] = 1 - full_df['EncodedPixels'].isna().astype(int)
train_df = full_df[full_df['ImageId'].isin(train_ids)]
valid_df = full_df[~full_df['ImageId'].isin(train_ids)]

train_unique_df = (train_df.groupby('ImageId')['target'].max() > 0).astype(int)

ids_0 = train_unique_df[train_unique_df == 0].index.values
ids_1 = train_df.loc[(train_df['ClassId'] == 1) & (train_df['target'] == 1), 'ImageId'].values
ids_2 = train_df.loc[(train_df['ClassId'] == 2) & (train_df['target'] == 1), 'ImageId'].values
ids_3 = train_df.loc[(train_df['ClassId'] == 3) & (train_df['target'] == 1), 'ImageId'].values
ids_4 = train_df.loc[(train_df['ClassId'] == 4) & (train_df['target'] == 1), 'ImageId'].values
ids_list = [ids_0 ,ids_1, ids_2, ids_3, ids_4]


def train(net, criterion, optimizer, scheduler, train_dataset, epoch, global_step):
    start_time = time.time()
    
    train_ldr = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, drop_last=True)
    total_loss_pxl_0 = 0
    total_loss_pxl_1 = 0
    total_loss_pxl_2 = 0
    total_loss_pxl_3 = 0
    total_loss_img   = 0
    tq = tqdm(enumerate(train_ldr), total=len(train_ldr))
    
    net.train()
    optimizer.zero_grad()
    for batch_idx, (image, mask, label) in tq:
        
        logit_pxl, logit_img = net(image)
        loss_pxl_0, loss_pxl_1, loss_pxl_2, loss_pxl_3, loss_img = criterion(logit_pxl, logit_img, mask, label)
        loss = loss_pxl_0 + loss_pxl_1 + loss_pxl_2 + loss_pxl_3 + loss_img * 10
        
        loss.backward()
            
        if (batch_idx + 1) % accumulation_steps == 0:             # Wait for several backward steps
            lr_this_step = lr * scheduler.get_lr(global_step, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)
            optimizer.step()                                      # Now we can do an optimizer step
            optimizer.zero_grad()
            
            global_step += 1
            
        total_loss_pxl_0 += loss_pxl_0.item()
        total_loss_pxl_1 += loss_pxl_1.item()
        total_loss_pxl_2 += loss_pxl_2.item()
        total_loss_pxl_3 += loss_pxl_3.item()
        total_loss_img   += loss_img.item()
        
        if batch_idx % LOG_INTERVAL == 0 and batch_idx > 0:
            cur_loss_pxl_0 = total_loss_pxl_0 / LOG_INTERVAL
            cur_loss_pxl_1 = total_loss_pxl_1 / LOG_INTERVAL
            cur_loss_pxl_2 = total_loss_pxl_2 / LOG_INTERVAL
            cur_loss_pxl_3 = total_loss_pxl_3 / LOG_INTERVAL
            cur_loss_img   = total_loss_img   / LOG_INTERVAL
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:3d}/{:3d} batches | ms/batch {:5.2f} | pxl loss 0 {:5.4f} | '
                'pxl loss 1 {:5.4f} | pxl loss 2 {:5.4f} | pxl loss 3 {:5.4f} | img loss {:5.4f} |'.format(
                epoch, batch_idx, len(train_ldr),
                elapsed * 1000 / LOG_INTERVAL, cur_loss_pxl_0, cur_loss_pxl_1, cur_loss_pxl_2, cur_loss_pxl_3, cur_loss_img))
            
            total_loss_pxl_0 = 0
            total_loss_pxl_1 = 0
            total_loss_pxl_2 = 0
            total_loss_pxl_3 = 0
            total_loss_img   = 0
            start_time = time.time()
    return global_step, lr_this_step


def dice_coef(preds, masks, thr=MASK_THR):
    """
    dice for threshold selection
    """
    n = preds.shape[0]
    c = preds.shape[1]
    
    masks = masks.view(n, -1)

    dices = torch.zeros(n, c)
    for i in range(c):
        preds_i = preds[:, i, :, :].view(n, -1)

        preds_m = preds_i > 0.5
        us = preds_m.sum(-1) < MASK_THR
        preds_m[us] = 0
        
        intersect = (preds_m * (masks==(i+1))).sum(-1).float()
        union = (preds_m.sum(-1) + (masks==(i+1)).sum(-1)).float()
        u0 = union==0
        intersect[u0] = 1
        union[u0] = 2
        dices[:, i] = (2. * intersect / union)
    return dices


def val_metrics(preds, truths, thr=0.5):

    c = preds.shape[1]   
    conf_mat = np.zeros((c, 4))
    fscores = np.zeros(4)

    for i in range(c):
        t = truths[:, i].astype(int)
        p = (preds[:, i] > thr).astype(int)
        conf_mat[i] = confusion_matrix(t, p).ravel()
        fscores[i] = fbeta_score(t, p, average='binary', beta=0.5)

    return conf_mat, fscores


def evaluate(net, valid_dataset, batch_size=32):
    
    valid_size = len(valid_dataset)
    valid_ldr = DataLoader(valid_dataset, batch_size, shuffle=False)
    
    net.eval()
    
    preds_img = np.zeros((valid_size, 4))
    truths_img = np.zeros((valid_size, 4))
    dice_mat = np.zeros((int(np.ceil(valid_size / batch_size)), 4))
    tq = tqdm(valid_ldr)
    with torch.no_grad():
        for batch_idx, (image, mask, label) in enumerate(tq):
            
            logit_pxl, logit_img = net(image)

            pred_img = torch.sigmoid(logit_img)
            preds_img[batch_idx*batch_size:(batch_idx+1)*batch_size] = pred_img.detach().cpu().numpy()
            truths_img[batch_idx*batch_size:(batch_idx+1)*batch_size] = label.detach().cpu().numpy()

            logit_pxl = F.interpolate(input=logit_pxl, size=tuple(mask.shape[1:]), mode='bilinear', align_corners=False)
            pred_pxl = torch.sigmoid(logit_pxl)
            pred_img = (pred_img > 0.5).float()
            pred_img[:, 2] = 1.
            pred_pxl = pred_pxl * pred_img[:, :, None, None]
            dice_mat[batch_idx, :] = dice_coef(pred_pxl, mask).sum(0)
            
    conf_mat, fscores = val_metrics(preds_img, truths_img, thr=0.5)
    dices = dice_mat.sum(axis=0) / valid_size
    return conf_mat, fscores, dices


def main():
    create_logging(LOGS_DIR, 'w')

    train_dataset = BalanceCropTrainSetCS(ids_list, train_df, DATA_DIR, sample_type=SAMPLE_TYPE)
    valid_dataset = ValSetCS(valid_ids, valid_df, DATA_DIR)
    
    net = EfficientUnet(encoder_type=MODEL_TYPE, dropout_p=0.5, attention_type='cbam', out_channel=4, imagenet_pretrained=True).to(DEVICE)
    
    num_train_optimization_steps = int(MAX_EPOCHS * len(train_dataset) / BATCH_SIZE / accumulation_steps)
    criterion = SeverstalClsSegLossV2()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = WarmupConstantSchedule(warmup=warmup_proportion,
                                       t_total=num_train_optimization_steps)
#     net, optimizer = amp.initialize(net, optimizer, opt_level="O2",
#                                     verbosity=0)
    
    best_so_far = 0
    global_step = 0
    for epoch in tqdm(range(0, STAGE_EPOCHS)):

        epoch_start_time = time.time()

        global_step, current_lr = train(net, criterion, optimizer, scheduler, train_dataset, epoch, global_step)
        logging.info("Current Learning rate: {:.5f}".format(current_lr))
        logging.info("iter 0: {} | iter 1: {} | iter 2: {} | iter 3: {} | iter 4: {}".format(*train_dataset.iter_list))
        
        if epoch % 10 == 9:
            conf_mat, fscores, val_dice = evaluate(net, valid_dataset, BATCH_SIZE)
            val_f = fscores.mean()
            val_score = val_dice.mean()

            logging.info('-' * 89)
            logging.info('end of epoch {:4d} | time: {:5.2f}s     | val dice {:5.4f}   | macro f {:5.4f}    |'.format(
                epoch, (time.time() - epoch_start_time), val_score, val_f))
            logging.info('val dice 1 {:5.4f} | val dice 2 {:5.4f} | val dice 3 {:5.4f} | val dice 4 {:5.4f} |'.format(
                val_dice[0], val_dice[1], val_dice[2], val_dice[3]))
            logging.info('tn 1 {:5.0f} | fp 1 {:5.0f} | fn 1 {:5.0f} | tp 1 {:5.0f} |'.format(
                conf_mat[0, 0], conf_mat[0, 1], conf_mat[0, 2], conf_mat[0, 3]))
            logging.info('tn 2 {:5.0f} | fp 2 {:5.0f} | fn 2 {:5.0f} | tp 2 {:5.0f} |'.format(
                conf_mat[1, 0], conf_mat[1, 1], conf_mat[1, 2], conf_mat[1, 3]))
            logging.info('tn 3 {:5.0f} | fp 3 {:5.0f} | fn 3 {:5.0f} | tp 3 {:5.0f} |'.format(
                conf_mat[2, 0], conf_mat[2, 1], conf_mat[2, 2], conf_mat[2, 3]))
            logging.info('tn 4 {:5.0f} | fp 4 {:5.0f} | fn 4 {:5.0f} | tp 4 {:5.0f} |'.format(
                conf_mat[3, 0], conf_mat[3, 1], conf_mat[3, 2], conf_mat[3, 3]))

            logging.info('-' * 89)
            
            if val_score > best_so_far:
                best_so_far = val_score
                logging.info('Higher dice achieved - saving model...')
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
    #                 'amp': amp.state_dict(),
                    'score': val_score,
                }, os.path.join(SAVE_PATH, 'unet_{}_light_cls_seg.pt'.format(MODEL_TYPE)))

    logging.info('Finish training - saving last checkpoint')
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
#         'amp': amp.state_dict(),
        'score': val_score,
    }, os.path.join(SAVE_PATH, 'unet_{}_light_cls_seg_last.pt'.format(MODEL_TYPE)))
    

if __name__ == "__main__":
	main()
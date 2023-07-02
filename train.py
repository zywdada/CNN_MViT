from efficientnet_pytorch import EfficientNet
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, choice
from pytorch_pretrained_vit import ViT
import numpy as np
from torch.optim import lr_scheduler
import os
import json
from os import cpu_count
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from efficient_vit import EfficientViT

from my_MViT_SE import mmvit_SE
from my_MViT_Cross import mmvit_cross
from my_Mvit import mmvit
from Mvit import Mvit
# from spectformer import spectformer_b
# from  newmodel  import Transformer
# from model_time import x_transformer
import uuid
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
import cv2
from transforms.albu import IsotropicResize
import glob
import pandas as pd
from tqdm import tqdm
from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
from sklearn.utils.class_weight import compute_class_weight 
from torch.optim.lr_scheduler import LambdaLR
import collections
from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse
from dataloader import MyIterableDataset

from cswin import UniNeXt_B
from beit import beit


# BASE_DIR = 'E:\\study\\data\\tttest'
BASE_DIR = 'E:\\study\\code\\vit\\efficient-vit'
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
MODELS_PATH = "E:\\study\\code\\mvit\\models\\beit"
METADATA_PATH = os.path.join(BASE_DIR, "data\\training_set_DFDC\\metadata03") # Folder containing all training metadata for DFDC dataset
VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels.csv")



def read_frames(video_path, train_dataset, validation_dataset, config):
    
    # Get the video label based on dataset selected
    # method = get_method(video_path, DATA_DIR)
    if TRAINING_DIR in video_path:
        if "Original" in video_path:
            label = 0.
        else:
            label = 1.
        if label == None:
            print("NOT FOUND", video_path)


    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    if label == 0:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing_real']),1) # Compensate unbalancing
    else:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing_fake']),1)

    
    # if VALIDATION_DIR in video_path:
    #     min_video_frames = int(max(min_video_frames/8, 2))
    frames_interval = int(frames_number / min_video_frames)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0,1):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)

    # Select only the frames at a certain interval
    if frames_interval > 0:
        for key in frames_paths_dict.keys():
            if len(frames_paths_dict) > frames_interval:
                frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
            
            frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]

    # Select N frames from the collected ones
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            #image = transform(np.asarray(cv2.imread(os.path.join(video_path, frame_image))))
            image = cv2.imread(os.path.join(video_path, frame_image))
            q = image.shape
            if image is not None:
                if TRAINING_DIR in video_path:
                    train_dataset.append((image, label))
                else:
                    validation_dataset.append((image, label))

# Main body
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=60, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='models\\beit\efficientnetB0_checkpoint1_All', type=str, metavar='PATH',
                        help='Path to latest checkpoint.')
    parser.add_argument('--dataset', type=str, default='All', 
                        help="Which dataset to use ")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, default='E:\\study\\code\\mvit\\configs\\architecture.yaml',
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--patience', type=int, default=3, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    
    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
 
    
    # model = mmvit_SE()
   # model = mmvit_cross()
    model = beit()
    # model = spectformer_b()
    # model = Transformer(192, 6, 16, 49, 1)
    # model.head = nn.Linear(512, 1)
    # print(model)
    # model = EfficientViT(config=config, channels=channels, selected_efficient_net = opt.efficient_net)
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume), strict=False)
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1 # The checkpoint's file name format should be "checkpoint_EPOCH"
    else:
        print("No checkpoint loaded.")

    
    # for name, param in model.named_parameters():
    #     if "Mvitv2" in name:
    #         param.requires_grad = False

    print("Model Parameters:", get_n_params(model))
    

   
    #READ DATASET
    if opt.dataset != "All" and opt.dataset != "DFDC":
        folders = ["Original", opt.dataset]
    else:
        folders = ["Original", "DFDC", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    folders = ["Deepfakes", 'Original']

    sets = [TRAINING_DIR, VALIDATION_DIR]

    train_paths = []
    validation_paths = []
    for dataset in sets:
        for folder in folders:
            subfolder = os.path.join(dataset, folder)
            if os.path.exists(subfolder):
                for index, video_folder_name in enumerate(os.listdir(subfolder)):
                    if index == opt.max_videos:
                        break
                    if os.path.isdir(os.path.join(subfolder, video_folder_name)):
                        if dataset == TRAINING_DIR:
                            train_paths.append(os.path.join(subfolder, video_folder_name))
                        else:
                            validation_paths.append(os.path.join(subfolder, video_folder_name))
                

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Create the data loaders
    # validation_labels = np.asarray([row[1] for row in validation_dataset])
    # labels = np.asarray([row[1] for row in train_dataset])


    train_dataset = MyIterableDataset(0, len(train_paths), train_paths, config)
    train_samples = train_dataset.__len__()
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    del train_dataset

    validation_dataset = MyIterableDataset(0, len(validation_paths), validation_paths, config)
    validation_samples = validation_dataset.__len__()
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=8, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    del validation_dataset
    

    model = model.cuda()
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*config['training']['bs'])+len(val_dl))
        train_correct = 0
        positive = 0
        negative = 0
        for index, (images, labels) in enumerate(dl):
            images = np.transpose(images, (0, 3, 1, 2))
            qq = images.shape
            labels = labels.unsqueeze(1)
            images = images.cuda()
            
            y_pred = model(images)
            y_pred = y_pred.cpu()
            
            loss = loss_fn(y_pred, labels)
        
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)
            
            if index%400 == 0: # Intermediate metrics print
                print("\nLoss: ", total_loss/counter, "Accuracy: ",train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive)


            for i in range(config['training']['bs']):
                bar.next()

        val_correct = 0
        val_positive = 0
        val_negative = 0
        val_counter = 0
        train_correct /= train_samples
        total_loss /= counter
        for index, (val_images, val_labels) in enumerate(val_dl):
    
            val_images = np.transpose(val_images, (0, 3, 1, 2))
            
            val_images = val_images.cuda()
            val_labels = val_labels.unsqueeze(1)
            val_pred = model(val_images)
            val_pred = val_pred.cpu()
            val_loss = loss_fn(val_pred, val_labels)
            total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
            val_correct += corrects
            val_positive += positive_class
            val_counter += 1
            val_negative += negative_class
            bar.next()
            
        scheduler.step()
        bar.finish()
            

        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0
        
        previous_loss = total_val_loss
        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
            str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct))
    
        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)
        print("--------save model---------")
        torch.save(model.state_dict(), os.path.join(MODELS_PATH,  "CNN_MViT"+"_checkpoint" + str(t) + "_" + opt.dataset))
        
        

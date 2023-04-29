
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
import requests
import time
import numpy as np
import io
from io import BytesIO
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence
import random
from tqdm import tqdm
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
import threading
import torchvision.models as models
import torch.nn as nn
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel
from nltk.corpus import wordnet
from image_transforms import SimCLRData_image_Transform
from dataset import FlickrDataset,Flickr30kDataset
from model_advml import ResNetSimCLR
from metrics import  intra_ContrastiveLoss ,Optimizer_simclr
from utils import get_gpu_stats,layerwise_trainable_parameters,count_trainable_parameters
from train_fns import train,val
from logger import Logger
from args import args_project


# In[2]:


def main(args):

    torch.manual_seed(1234)
    trial_number=args.trial_number
    flickr30k_images_dir_path=args.flickr30k_images_dir_path
    flickr30k_tokens_dir_path=args.flickr30k_tokens_dir_path
    logresults_save_dir_path = '/work/08629/pradhakr/maverick2/advml_project/train_results'
    train_log = os.path.join(logresults_save_dir_path, f'train{trial_number}_30k.log')
    image_model_log = os.path.join(logresults_save_dir_path, f'image_model{trial_number}_30k.pth')
    graph_save_dir=args.graph_save_dir
    batch_size=args.batch_size
    intra_projection_dim=args.intra_projection_dim
    hidden_dim =args.hidden_dim
    layers_to_train=args.image_layers_to_train
    optimizer_name=args.optimizer_type
    lr=args.image_learning_rate
    momentum=args.momentum
    weight_decay=args.weight_decay
    reconstruction_tradeoff=args.reconstruction_tradeoff
    contrastive_tradeoff=args.contrastive_tradeoff
    temperature=args.temperature
    total_epochs=args.total_epochs
    noise_amount=args.noise_amount
    scheduler_image=None
    dataset = Flickr30kDataset(flickr30k_images_dir_path, 
                               flickr30k_tokens_dir_path,
                               caption_index_1=0,
                               caption_index_2=1,
                              image_transform=SimCLRData_image_Transform(),
                              noise_transform=True,
                                  evaluate=False,
                                  noise_amount=noise_amount)
    indices = list(range(len(dataset)))
    train_indices = indices[:29783]
    val_indices = indices[29783:30783]
    test_indices = indices[30783:]
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    batch_size=batch_size
    train_loader = DataLoader(train_set, 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 num_workers=4, 
                                 pin_memory=True)
    val_loader = DataLoader(val_set, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=4, 
                             pin_memory=True)
    test_loader = DataLoader(test_set, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=4, 
                             pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = ResNetSimCLR(
            model='resnet50',
            intra_projection_dim=intra_projection_dim,
            hidden_dim =hidden_dim,
            layers_to_train=layers_to_train
        ).to(device)
    optimizer_image = Optimizer_simclr(optimizer_name=optimizer_name,
                                           model_parameters=resnet_model.parameters(),
                                           lr=lr,
                                           momentum=momentum,
                                           weight_decay=weight_decay)
    optimizer_image = optimizer_image.optimizer
    #scheduler_image = optimizer_image.scheduler

    intra_loss=intra_ContrastiveLoss(device,temperature=temperature)
    logger_save = Logger(train_log,
                         image_model_log,
                         optimizer_name, 
                         lr,
                         weight_decay,
                         batch_size,
                         momentum, 
                         temperature, 
                         total_epochs,
                         reconstruction_tradeoff,
                         contrastive_tradeoff,
                         intra_projection_dim,
                         hidden_dim,
                        scheduler_image,
                         layers_to_train,
                         noise_amount
                         )
    logger_save.start_training()
        # Loop through epochs and train the models
    for epoch in tqdm(range(total_epochs)):

        start = time.time()
        train_loss=train(train_loader,
              resnet_model, 
              optimizer_image,
              intra_loss,
               device,
              scheduler_image ,
              reconstruction_tradeoff,
              contrastive_tradeoff)
        test_loss=val(val_loader,
            resnet_model,
            intra_loss,
            device,
            reconstruction_tradeoff,
            contrastive_tradeoff)
        end = time.time()

        # Log the results of the epoch
        logger_save.log(epoch + 1, resnet_model, train_loss, test_loss, end - start)
    logger_save.end_training()
    print('training_end')
    logger_save.plot_losses(trial_number,
                            graph_save_dir
                            )
if __name__ == '__main__':
    # Parse command-line arguments
    args = args_project()

    # Call the main function with the parsed arguments
    main(args)



# coding: utf-8

# In[5]:


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
torch.cuda.empty_cache()
torch.manual_seed(1234)


def train(dataloader, image_model, optimizer, intra_criterion,device,
          scheduler_image=None, reconstruction_tradeoff=1,contrastive_tradeoff=0):
    loss_epoch = 0
    mse_loss = nn.MSELoss()
    for idx, batch in enumerate(dataloader):
        image_model.train()
        batch_size = batch[0].shape[0]
        original_images,noisy_images,contrastive_img1,contrastive_img2=batch[0],batch[1],batch[2],batch[3]
        
        
        const_embeddings1 = image_model(contrastive_img1, device ,reconstruct=False,contrast=True)
        const_embeddings2 = image_model(contrastive_img2, device ,reconstruct=False,contrast=True)
        
        
        contrastive_loss=contrastive_tradeoff * intra_criterion(const_embeddings1, const_embeddings2, batch_size) 
        del const_embeddings1,const_embeddings2
        noisy_reconst_images = image_model(noisy_images, device ,reconstruct=True,contrast=False)

        reconstructive_loss= reconstruction_tradeoff*mse_loss(original_images.to(device),noisy_reconst_images)
        del noisy_reconst_images,original_images
        total_loss = contrastive_loss + reconstructive_loss
        total_loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        loss_epoch += total_loss.item()
    if scheduler_image:
        scheduler_image.step()
    epoch_loss = loss_epoch / len(dataloader)
    return epoch_loss
def psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val ** 2 / mse)
def val(dataloader, image_model, intra_criterion,device,
         reconstruction_tradeoff=1,contrastive_tradeoff=1):
    loss_epoch = 0
    mse_loss = nn.MSELoss()
    for idx, batch in enumerate(dataloader):
        image_model.eval()
        batch_size = batch[0].shape[0]
        original_images,noisy_images,contrastive_img1,contrastive_img2=batch[0],batch[1],batch[2],batch[3]
        
        
        const_embeddings1 = image_model(contrastive_img1, device ,reconstruct=False,contrast=True)
        const_embeddings2 = image_model(contrastive_img2, device ,reconstruct=False,contrast=True)
        
        
        contrastive_loss=contrastive_tradeoff * intra_criterion(const_embeddings1, const_embeddings2, batch_size) 
        del const_embeddings1,const_embeddings2
        noisy_reconst_images = image_model(noisy_images, device ,reconstruct=True,contrast=False)

        reconstructive_loss= reconstruction_tradeoff*mse_loss(original_images.to(device),noisy_reconst_images)
        del noisy_reconst_images,original_images

        total_loss = contrastive_loss + reconstructive_loss

        loss_epoch += total_loss.item()
    epoch_loss = loss_epoch / len(dataloader)
    return epoch_loss
def calculate_psnr(dataloader, image_model,device):
    loss_epoch = 0
    for idx, batch in enumerate(dataloader):
        image_model.eval()
        batch_size = batch[0].shape[0]
        original_images,noisy_images,contrastive_img1,contrastive_img2=batch[0],batch[1],batch[2],batch[3]
        
        
        noisy_reconst_images = image_model(noisy_images, device ,reconstruct=True,contrast=False)
        

        psnr_loss= psnr(original_images.to(device),noisy_reconst_images)
        del noisy_reconst_images,original_images


        loss_epoch += psnr_loss.item()
    epoch_loss = loss_epoch / len(dataloader)
    return epoch_loss
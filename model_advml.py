# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

class ResNetSimCLR(nn.Module):
    def __init__(self, model='resnet50', intra_projection_dim=128,hidden_dim=512,
                 layers_to_train=['layer4']):
        """
        Initializes ResNetSimCLR model.

        Parameters:
        - model: str, the ResNet model to use (default: 'resnet18')
        - projection_dim: int, the dimension of the projection head output (default: 128)
        - layers_to_train: list of str, the names of the layers in the ResNet model to train (default: ['layer4'])
        """
        super(ResNetSimCLR, self).__init__()

        # Instantiate the backbone ResNet model
        if model == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            in_features = 512
        elif model == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            in_features = 2048
        elif model == 'resnet101':
            backbone = models.resnet101(pretrained=True)
            in_features = 2048
        else:
            raise ValueError('Unsupported ResNet model:', model)

        # Freeze the layers that are not specified in layers_to_train
        for name, child in backbone.named_children():
            if name not in layers_to_train:
                for param in child.parameters():
                    param.requires_grad = False

        # Remove last fully-connected layer from the backbone
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # Define the transform to be applied to input images
        self.transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
        projection_head_input=in_features
        # Add the projection head layers
        self.contrastive_head = nn.Sequential(
            nn.Linear(projection_head_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, intra_projection_dim)
        )
        self.reconstruction_head = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )
        


    def forward(self, x, device , reconstruct=False , contrast=False):
        """
        Performs a forward pass through the ResNetSimCLR model.

        Parameters:
        - x: tensor of shape (batch_size, 3, height, width), the input images

        Returns:
        - features: tensor of shape (batch_size, in_features), the features extracted from the backbone
        - projection: tensor of shape (batch_size, projection_dim), the projections of the features
        """
        # Apply the transform to the input images
        #x = self.transform(x)
        x=x.to(device)

        # Extract features from the backbone
        #x = x.unsqueeze(0) # Add batch dimension
        x= self.backbone(x)
        if reconstruct:
            reconstruct_output= self.reconstruction_head(x)
            return reconstruct_output
        elif contrast:
            avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
            x1=avgpool(x)
            features = x1.view(x1.size(0), -1)
            embeddings=  self.contrastive_head(features)
            return embeddings
    
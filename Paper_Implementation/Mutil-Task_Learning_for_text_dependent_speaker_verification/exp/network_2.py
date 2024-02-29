import torch
from tqdm.auto import tqdm
import torch.nn as nn # 신경망들이 포함됨
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU 할당

import torchvision.models


# print(model(torch.rand(10, 1, 40, 42).to(device)))
class CNNBinaryClassification(torch.nn.Module):
    def __init__(self, n_class, dropout_prob=0.5):
        self
        super(CNNBinaryClassification, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=2, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(10, 100, kernel_size=2, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=2, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(200, 300, kernel_size=2, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.dropout = nn.Dropout(dropout_prob)



        self.fc_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(300, n_class), #fully connected layer(ouput layer)
            nn.Sigmoid()
        )    
        
    def forward(self, x):
        
        x = self.layer1(x) #1층
        
        x = self.layer2(x) #2층
        
        x = self.layer3(x) #3층
        
        x = self.layer4(x) #4층
        # N, 300, a, b  --> N, 300 x a x b
        # N, 300, a, b --> N, 300, 1, 1 --> N, 300
        #         
        #x = torch.flatten(x, start_dim=1) # N차원 배열 -> 1차원 배열

        x = self.dropout(x)  # 드롭아웃 적용


    
        out = self.fc_layer(x)
        return out
    



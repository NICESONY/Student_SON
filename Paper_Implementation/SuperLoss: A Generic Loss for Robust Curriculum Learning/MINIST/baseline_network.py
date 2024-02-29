import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, n_class):
        super(CNN, self).__init__() ## sequential를 쓰는 것을 권장한다.
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=2, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(100, 400, kernel_size=2, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(400, 800, kernel_size=2, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(800, 1000, kernel_size=2, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
 

        self.fc_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1000, n_class), #fully connected layer(ouput layer)
            #nn.Softmax()
        )    
        
    def forward(self, x):
        
        x = self.layer1(x) #1층
        x = self.layer2(x) #2층
        x = self.layer3(x) #3층
        x = self.layer4(x) #4층
        out = self.fc_layer(x)
        return out
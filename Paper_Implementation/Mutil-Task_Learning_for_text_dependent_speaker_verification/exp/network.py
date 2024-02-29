import torch
import torch.nn as nn  # 신경망들이 포함됨


class MultiTaskCNN(torch.nn.Module):
    def __init__(self, n_class_task1, n_class_task2):
        super(MultiTaskCNN, self).__init__()
        
        # Shared convolutional layers
        self.shared_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 100, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(100, 200, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(200, 300, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()

        )
        
        # Task-specific layers
        self.task1_layers = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(),
            nn.Linear(300, n_class_task1)
        )
        
        self.task2_layers = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(),
            nn.Linear(300, n_class_task2)
        )

    def forward(self, x):
        shared_out = self.shared_layers(x)
        task1_out = self.task1_layers(shared_out)
        task2_out = self.task2_layers(shared_out)
        return task1_out, task2_out




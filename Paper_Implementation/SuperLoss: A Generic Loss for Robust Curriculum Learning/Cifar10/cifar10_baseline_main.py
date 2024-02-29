import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from cifar10_code.baselin_eval import 



def prepare_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    data_root = "./data"

    train_set_all = datasets.CIFAR10(root = data_root, train = True, download = True, transform = transform) 
    test_set = datasets.CIFAR10(root = data_root, train = False, download = True, transform = transform)


    train_size = len(train_set_all)
    dev_size = int(0.2 * train_size)


    idxes = list(range(train_size))
    np.random.shuffle(idxes)


    train_set_idx = idxes[dev_size :]
    dev_set_idx = idxes[:dev_size]


    train_set = Subset(train_set_all, train_set_idx)
    dev_set = Subset(train_set_all, dev_set_idx)


    batch_size = 128
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True) 
    dev_loader = DataLoader(dev_set, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
    return train_loader, dev_loader ,test_loader


def get_image_from_dataloader(data_loader):
    for images, labels in data_loader:
        break
    print(images.shape)
    return images, labels


class CNN(nn.Module):
    def __init__(self, n_class):
        super(CNN, self).__init__() ## sequential를 쓰는 것을 권장한다.
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=2, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)) 
        
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(100, 400, kernel_size=2, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)) 
        
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(400, 800, kernel_size=2, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)) 
        
        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(800, 1000, kernel_size=2, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)) 
        
        self.fc_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1000, n_class), 
            
        )    
        
    def forward(self, x):
        x = self.layer1(x) #1층
        x = self.layer2(x) #2층
        x = self.layer3(x) #3층
        x = self.layer4(x) #4층
        out = self.fc_layer(x)
        return out





def train(net, optimizer, criterion, train_loader, device):

    train_loss = 0
    train_acc = 0
    count = 0

    net.train() 
    for inputs, labels in train_loader : ## feature과 label을 넣는다.
        count += len(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.item() 

        loss.backward()
        optimizer.step()
        predicted = torch.max(outputs, 1)[1]
        train_acc += (predicted == labels).sum().item() ## 평균을 하기 위해서는 배치사이즈로 나누어야함

    avg_train_loss = train_loss / count
    avg_train_acc = train_acc / count

    return  avg_train_loss, avg_train_acc


def test(net,  criterion,  test_loader, device) :
    net.eval()
    count = 0

    val_acc = 0
    val_loss = 0

    for inputs, labels in test_loader:
        count += len(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()



        predicted = torch.max(outputs, 1)[1]
        val_acc += (predicted == labels).sum().item()

    avg_val_loss = val_loss / count
    avg_val_acc = val_acc / count

    return avg_val_loss, avg_val_acc


def torch_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_determinstic_algoriths = True


def evaluate_history(history):
    print(f"inital state val_loss : {history[0,2]:.5f}"
          ,f"inital stast val_acc :  {history[0,4] :.5f}")
    print(f"final state val_loss : {history[-1,2]:.5f}"
          ,f"final stast val_acc :  {history[-1,4] :.5f}")

    num_epochs = len(history)
    unit = num_epochs / 10
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], "b",label = "train")
    plt.plot(history[:, 0], history[:, 3], "k", label="val")
    plt.xticks(np.arange(0, num_epochs+1, unit))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("learning curve(loss)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(9,8))
    plt.plot(history[:, 0], history[:,2], "b", label = "train")
    plt.plot(history[:, 0], history[:, 4], "k", label = "val")
    plt.xticks(np.arange(0, num_epochs+1 , unit))
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("learning curve(acc)")
    plt.legend()
    plt.show()



def main():
    device = torch.device("cuda:0" if torch .cuda.is_available() else "CPU")
    print(device)

    train_loader, dev_loader, test_loader = prepare_dataset()
    


    torch_seed()
    net = CNN(n_class= 10).to(device)
    criterion = nn.CrossEntropyLoss()

    lr = 0.01
    optimizer = optim.RAdam(net.parameters(), lr =lr)
    num_epoch = 50
    history = np.zeros((0,5))

    base_epoch = len(history)
    for epoch in range(base_epoch, num_epoch+base_epoch):
        avg_train_loss, avg_train_acc = train(net, optimizer, criterion, train_loader, device)
        avg_val_loss, avg_val_acc = test(net, criterion, test_loader, device)
        avg_dev_loss, avg_dev_acc = test(net, criterion, dev_loader, device)

        print(f"epoch [{epoch +1}/{num_epoch+base_epoch}]", 
              f"train_loss:{avg_train_loss:.5f}, train_acc: {avg_train_acc :.5f}",
              f"val_loss : {avg_val_loss :.5f}, val_acc : {avg_val_acc : .5f}",
              f"dev_loss : {avg_dev_loss :.5f}, dev_acc : {avg_dev_acc : .5f}")
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))


    evaluate_history(history)




if __name__ == "__main__" :
    main()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from MINIST.baseline_costomdataset import prepare_dataset
from MINIST.baseline_eval import test
from MINIST.baseline_network import CNN
from MINIST.baseline_train import train
from MINIST.baseline_utils import torch_seed, evaluate_history
from MINIST.resnet18 import ResNet ,resnet18, resnet34





def main():
    device = torch.device("cuda:0" if torch .cuda.is_available() else "CPU")
    print(device)

    train_loader, dev_loader, test_loader = prepare_dataset()
    


    torch_seed()
    #net = CNN(n_class= 10).to(device)
    net = resnet34(num_classes =100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(net.parameters(), lr= 0.001, weight_decay= 1e-5)
    num_epoch = 100
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

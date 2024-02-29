import torch
import torch.nn as nn


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



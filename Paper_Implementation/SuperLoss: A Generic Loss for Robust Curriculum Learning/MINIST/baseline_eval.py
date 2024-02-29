import torch



def test(net, criterion, test_loader, device) :
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

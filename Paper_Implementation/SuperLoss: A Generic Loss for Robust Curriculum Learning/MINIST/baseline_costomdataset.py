import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
import torchvision.datasets as datasets
import numpy as np




def prepare_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    data_root = "/media/syoon/DATA2/2024_WINTTER_RAB/baseline_code/MINIST/data"

    train_set_all = datasets.MNIST(root = data_root, train = True, download = True, transform = transform) 
    test_set = datasets.MNIST(root = data_root, train = False, download = True, transform = transform)


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
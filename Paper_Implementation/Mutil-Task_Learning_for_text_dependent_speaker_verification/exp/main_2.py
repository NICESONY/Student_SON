import torch
import numpy as np
import pandas as pd
import librosa.display as dsp
from IPython.display import Audio
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import torchvision.datasets as datasets # 데이터셋 집합체
import torchvision.transforms as transforms # 변환 툴
from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴
from lib.util_2 import get_argparser, load_dataset, set_length, preprocess_dataset
from lib.evaluate_2 import predict_eval_module
from lib.dataset_2 import CustomDataset
from lib.network_2 import CNNBinaryClassification
from lib.train_2 import train
import argparse




# 에포크 설정
#num_epochs = 10
# 배치 사이즈 설정
#batch_size = 100


#target_length = 21000



#train_scp = 'E:/data/audio_mnist/wav_audio_mnist_train.scp' # 경로 이 부분 배웠다. 다시 복습
#dev_scp = 'E:/data/audio_mnist/wav_audio_mnist_dev.scp' # 경로 이 부분 배웠다. 다시 복습
#eval_scp = 'E:/data/audio_mnist/wav_audio_mnist_eval.scp'# 경로 이 부분 배웠다. 다시 복습




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU 할당







def main(args: argparse.Namespace) -> None:
    # global device ,num_epochs, batch_size

    train_scp = f'{args.base_dir}/scp/wav_audio_mnist_train_2_modified.scp' # 경로 이 부분 배웠다. 다시 복습
    dev_scp = f'{args.base_dir}/scp/wav_audio_mnist_dev_2_modified.scp' # 경로 이 부분 배웠다. 다시 복습
    eval_scp = f'{args.base_dir}/scp/wav_audio_mnist_eval_2_modified.scp'# 경로 이 부분 배웠다. 다시 복습


    train_wav, train_ids = load_dataset(train_scp)
    # print(train_wav)
    # print(train_ids)
    # exit()

    eval_wav, eval_ids = load_dataset(eval_scp)

    dev_wav, dev_ids = load_dataset(dev_scp)



    train_wav = set_length(train_wav, args.target_length)
    dev_wav = set_length(dev_wav, args.target_length)
    eval_wav = set_length(eval_wav, args.target_length)

    print('train :', train_wav.shape)
    print('dev :', dev_wav.shape)
    print('eval :', eval_wav.shape)
    # exit()



        
    
    train_mfccs = preprocess_dataset(train_wav)
    # train_mfccs = train_mfccs.reshape(-1, train_mfccs.shape[1], train_mfccs.shape[2], 1)


    dev_mfccs = preprocess_dataset(dev_wav)
    # dev_mfccs = dev_mfccs.reshape(-1, dev_mfccs.shape[1], dev_mfccs.shape[2], 1) # 2D CNN 넣기 위해서 하는 것

    eval_mfccs = preprocess_dataset(eval_wav)
    # eval_mfccs = eval_mfccs.reshape(-1, eval_mfccs.shape[1], eval_mfccs.shape[2], 1)
    print(eval_mfccs.shape)
    

    """
    # Alternative example
    key2mfcc_train = {}
    """



    #만든 train dataset를 DataLoader에 넣어 batch 만들기
    train_dataset = CustomDataset(X = train_mfccs, y = train_ids)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size_train, shuffle=True)


    dev_dataset = CustomDataset(X = dev_mfccs, y = dev_ids)
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size_test, shuffle=False)
    
    eval_dataset = CustomDataset(X = eval_mfccs, y = eval_ids)
    eval_loader = DataLoader(eval_dataset, batch_size = args.batch_size_test, shuffle=False)
    
    



    train_batches = len(train_loader)
    dev_batches = len(dev_loader)
    eval_batches = len(eval_loader)

    print('/ total train batches :', train_batches)
    print('/ total valid batches :', dev_batches)
    print('/ total valid batches :', eval_batches)



    
    model = CNNBinaryClassification(args.n_classes).to(device)  
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    #optimizer = torch.optim.SGD(params = model.parameters(), lr = 1e-3 )
    optimizer = torch.optim.RAdam(params = model.parameters(), lr = args.lr)

    scheduler = None
    # exit()

    print(model(torch.rand(10, 1, 40, 42).to(device)))
    #exit()

    for epoch in range(1, args.n_epochs + 1):
        train(model, optimizer, train_loader, scheduler, device, epoch) 
        

        acc_dev = predict_eval_module(model, dev_loader, device)   # dev
        acc_eval = predict_eval_module(model, eval_loader, device)   # eval

        print()
        
        torch.save(model.state_dict(), f'E:/exp/audio_mnist_test/models/{epoch}')

        with open('E:/exp/audio_mnist_test/log.txt', 'a') as f:
            f.write(f'{epoch}\t{acc_dev}\t{acc_eval}\n')




if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    main(args)




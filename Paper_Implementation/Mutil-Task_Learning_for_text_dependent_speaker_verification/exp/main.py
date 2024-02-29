import torch
from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴
from lib.util import mfcc_read_srp, get_argparser, load_dataset, set_length, preprocess_dataset
from lib.evaluate import predict_eval_module
from lib.dataset import CustomDataset
from lib.network import MultiTaskCNN
from lib.train import train
import argparse
from lib.resnet import resnet18
import numpy as np





# 에포크 설정
#num_epochs = 10
# 배치 사이즈 설정
#batch_size = 100


#target_length = 21000
target_length = 64



#train_scp = 'E:/data/audio_mnist/wav_audio_mnist_train.scp' # 경로 이 부분 배웠다. 다시 복습
#dev_scp = 'E:/data/audio_mnist/wav_audio_mnist_dev.scp' # 경로 이 부분 배웠다. 다시 복습
#eval_scp = 'E:/data/audio_mnist/wav_audio_mnist_eval.scp'# 경로 이 부분 배웠다. 다시 복습




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU 할당



def get_key2txt(key: str) -> int:
    # {txt}_{spk}_{sess}
    return int(key.split('_')[0])





def main(args: argparse.Namespace) -> None:
    # global device ,num_epochs, batch_size

    """
    train_scp = f'{args.base_dir}/scp/wav_audio_mnist_train.scp' # 경로 이 부분 배웠다. 다시 복습
    dev_scp = f'{args.base_dir}/scp/wav_audio_mnist_dev.scp' # 경로 이 부분 배웠다. 다시 복습
    eval_scp = f'{args.base_dir}/scp/wav_audio_mnist_eval.scp'# 경로 이 부분 배웠다. 다시 복습


    train_wav, train_ids = load_dataset(train_scp)
    # print(train_wav)
    # print(train_ids)
    # exit()

    eval_wav, eval_ids = load_dataset(eval_scp)

    dev_wav, dev_ids = load_dataset(dev_scp)



    # train_wav = set_length(train_wav, args.target_length)
    # dev_wav = set_length(dev_wav, args.target_length)
    # eval_wav = set_length(eval_wav, args.target_length)

    # print('train :', train_wav.shape)
    # print('dev :', dev_wav.shape)
    # print('eval :', eval_wav.shape)       
    # exit()



    """
    # MFCC 데이터가 저장된 폴더 경로
    mfcc40_scp = 'E:/정리/data/audio_mnist/scp/mfcc40_{}.scp'
    


    train_mfccs = mfcc_read_srp(mfcc40_scp.format("train"))
    dev_mfccs = mfcc_read_srp(mfcc40_scp.format("dev"))
    eval_mfccs = mfcc_read_srp(mfcc40_scp.format("eval"))
    
    '''
    64.76473 11.118968 30.0 101.0 64.0
    lengths = []
    for v in train_mfccs.values():
        lengths.append(v.shape[-1])
    for v in dev_mfccs.values():
        lengths.append(v.shape[-1])
    for v in eval_mfccs.values():
        lengths.append(v.shape[-1])
    lengths = np.asarray(lengths, dtype=np.float32)
    print(lengths.mean(), lengths.std(), lengths.min(), lengths.max(), np.median(lengths))   
    '''
    
    
    # train_mfccs = preprocess_dataset(train_wav)
    # # train_mfccs = train_mfccs.reshape(-1, train_mfccs.shape[1], train_mfccs.shape[2], 1)


    # dev_mfccs = preprocess_dataset(dev_wav)
    # # dev_mfccs = dev_mfccs.reshape(-1, dev_mfccs.shape[1], dev_mfccs.shape[2], 1) # 2D CNN 넣기 위해서 하는 것

    # eval_mfccs = preprocess_dataset(eval_wav)
    # # eval_mfccs = eval_mfccs.reshape(-1, eval_mfccs.shape[1], eval_mfccs.shape[2], 1)
    # print(eval_mfccs.shape)
    



    #만든 train dataset를 DataLoader에 넣어 batch 만들기
    train_dataset = CustomDataset(key2mfcc = train_mfccs, target_length=target_length, label_fn=get_key2txt)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size_train, shuffle=True)

    dev_dataset = CustomDataset(key2mfcc= dev_mfccs, target_length=target_length, label_fn=get_key2txt)
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size_test, shuffle=False)
    
    eval_dataset = CustomDataset(key2mfcc = eval_mfccs, target_length=target_length, label_fn=get_key2txt)
    eval_loader = DataLoader(eval_dataset, batch_size = args.batch_size_test, shuffle=False)
    
    



    train_batches = len(train_loader)
    dev_batches = len(dev_loader)
    eval_batches = len(eval_loader)

    print('/ total train batches :', train_batches)
    print('/ total valid batches :', dev_batches)
    print('/ total valid batches :', eval_batches)



    
    #model = CNNclassification(args.n_classes).to(device)  
    # model = resnet18(args.n_classes).to(device)  
    model = MultiTaskCNN(args.n_class_task1, args.n_class_task2).to(device)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    #optimizer = torch.optim.SGD(params = model.parameters(), lr = 1e-3 )
    optimizer = torch.optim.RAdam(params = model.parameters(), lr = args.lr)

    # exit()

    print(model(torch.rand(10, 1, 40, 42).to(device)))
    # exit()

    for epoch in range(1, args.n_epochs + 1):
        train(model, optimizer, train_loader, device, epoch) # , device


        acc_dev = predict_eval_module(model, dev_loader,device)   # dev
        acc_eval = predict_eval_module(model, eval_loader,device)   # eval

        print()

        torch.save(model.state_dict(), f'E:/정리/exp/exp_1/audio_mnist_test/models{epoch}')

        with open('E:/정리/exp/exp_1/audio_mnist_test/log.txt', 'a') as f:
            f.write(f'{epoch}\t{acc_dev}\t{acc_eval}\n')




if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    main(args)




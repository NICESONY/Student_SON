from tqdm.auto import tqdm
import torch
import os
import pandas as pd
import numpy as np
import tqdm
import librosa
import pandas as pd
import librosa
import librosa.display as dsp
from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import torchvision.datasets as datasets # 데이터셋 집합체
import torchvision.transforms as transforms # 변환 툴
from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU 할당


    
# 에포크 설정
# num_epochs = 10
# 배치 사이즈 설정
# batch_size = 100  # 여기서 질문 :  여기서 에포크랑 밴치를 주는 게 맞을까? 아니면 어떻게 해야할지??  : 스스로가 찾은 내 대답은 여기가 맞음


#criterion = torch.nn.CrossEntropyLoss().to(device)



def train(model, optimizer, train_loader, scheduler, device, epoch):
    # global criterion # , num_epochs
    
    
    best_acc = 0
    
    # 에포크 설정
    model.train()  # 모델 학습
    #running_loss = 0.0
            
    for b, (wav, label) in enumerate(train_loader):

        wav = wav.to(device)
        
        label = torch.tensor([int(x) for x in label]).to(device) # 데이터를 GPU로 이동
        label = label.view(-1, 1)
        # torch.tensor([int(x) for x in label])
        
        # print(label.shape)
        optimizer.zero_grad()  # 배치마다 optimizer 초기화

        # Data -> Model -> Output
        logit = model(wav)  # 예측값 산출
        
        #loss = criterion(logit, label)  # 손실함수 계산
        # loss = torch.nn.functional.cross_entropy(logit, label)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label.float())

        
        # 역전파
        loss.backward()  # 손실함수 기준 역전파 
        optimizer.step()  # 가중치 최적화
        #running_loss += loss.item()

            
        #print('[%d] Train loss: %.10f' % (epoch+1, running_loss / len(train_loader)))
        print(f'\rEpoch {epoch:3d}\tloss: {loss.item():.4f}\t{b+1:3d} / {len(train_loader):3d}', end='\t')
        
        if scheduler is not None:
            scheduler.step()
    """
    # Validation set 평가
    model.eval()  # evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
    val_loss = 0.0
    correct = 0
    
    with torch.no_grad():  # 파라미터 업데이트 안하기 때문에 no_grad 사용
        for wav, label in dev_loader:
            wav, label = wav.to(device), torch.tensor([int(x) for x in label]).to(device)
            logit = model(wav)
            val_loss += criterion(logit, label)
            pred = logit.argmax(dim=1, keepdim=True)  # 10개의 class 중 가장 값이 높은 것을 예측 label로 추출
            correct += pred.eq(label.view_as(pred)).sum().item()  # 예측값과 실제값이 맞으면 1, 아니면 0으로 합산
            
    val_acc = 100 * correct / len(dev_loader.dataset)
    print('dev set: Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss / len(dev_loader), correct, len(dev_loader.dataset), val_acc))
    
    # Best 모델 저장
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print('Best model saved.')
    """

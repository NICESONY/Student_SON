import torch
import tqdm
import librosa
import librosa.display as dsp
from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os




import torch

def predict_eval_module(model, test_loader,device):
    model.eval()  # 모델을 평가 모드로 설정
    #total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 그라디언트 계산 비활성화
        for wav, labels in test_loader:
            wav = wav.to(device)
            labels = torch.tensor([int(i) for i in labels]).to(device)
            # labels = labels.view(-1, 1)
            logit = model(wav)
            _, predicted = torch.max(logit.data, 1)

            #total_loss += torch.nn.functional.cross_entropy(logit, labels, reduction='sum').item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    #avg_loss = total_loss / total
    accuracy = correct / total

    # return avg_loss , accuracy

    #print(f"평균 손실: {avg_loss:.4f}")
    print(f"accuracy: {accuracy:.4f}", end='\t')


# def predict_eval_module(model, eva_loader):
#     model.eval()
#     model_pred = []
#     with torch.no_grad():
#         for wav in eva_loader:
#             # wav = wav
#             wav = wav  # 테스트 해보자 이렇게 만들면 to 사용할 수 있다고 함.
#             # 여기서 라벨도 같이 불러오는 것이 맞을까? 아닐까? 
#             pred_logit = model(wav)
#             pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

#             model_pred.extend(pred_logit.tolist())
#     return model_pred
#     # 다음으로 바꿀 것은 y값을 None으로 주고 해보자 => 실패  
    
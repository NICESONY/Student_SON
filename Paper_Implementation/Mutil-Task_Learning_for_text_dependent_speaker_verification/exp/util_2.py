import librosa
import librosa.display as dsp
from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# from tqdm import tqdm
import argparse


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-03)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--batch_size_train', type=int, default=32)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--target_length', type=int, default=21000)
    #parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--base_dir', type=str, default='E:/data/audio_mnist')
    #....

    return parser



def load_dataset(scp_path):
    dataset = []
    ids = []
    with open(scp_path, 'r') as f:
        for line in f.readlines():
            id, wav_path = line.strip().split()
            data, sr = librosa.load(wav_path, sr=16000)
            class_label = id.split("_")[1]
            dataset.append(data)
            ids.append(class_label)
    return dataset , ids





def set_length(data : list, d_mini):
    'asfd' # 복습할 것
    result = []
    for wav_sample in data:
        # assert len(wav_sample) >= d_mini  ### 이 문장을 통해서 길이가 더 짥은게 있는지 확인할 수 있었다.
        if len(wav_sample) >= d_mini :
            result.append(wav_sample[:d_mini])
        else :
            num_pad = d_mini - len(wav_sample)
            wav_pad = np.pad(wav_sample, [0, num_pad], mode = "wrap")
            result.append(wav_pad)
    result = np.array(result)

    return result 




def preprocess_dataset(data, sr = 16000, n_mfcc = 40, hop_length = 160, win_length = 400, n_fft = 512, window='hamming'):
    mfccs = []
    for wav_sample in data:
        extracted_features = librosa.feature.mfcc(
            y = wav_sample, 
            sr = sr, 
            n_mfcc = n_mfcc,
            hop_length = hop_length,
            win_length = win_length,
            n_fft = n_fft,
            window=window
            )
        mfccs.append(extracted_features)
    mfccs = np.array(mfccs) # 복습 여기에 이 코드를 넣으므로써 main에서 구현 안해도 넘파이 배열 형식로 구현 가능
            
    return mfccs





# def load_dataset(scp_path: str, sr: int = 16000, n_mfcc: int = 40) -> dict:
#     # n_fft: int = 512, win_length: int = 400, hop_length: int = 160, window: str = 'hamming'
    
#     # Parameters:
#     #     scp_path (str): 데이터셋 파일의 경로
#     #     sr (int): 샘플링 레이트 (Sample rate)
#     #     n_mfcc (int): 추출할 MFCC (Mel-frequency cepstral coefficients)의 개수
#     #     n_fft (int): FFT (Fast Fourier Transform)의 윈도우 크기
#     #     win_length (int): 스펙트로그램 계산에 사용되는 윈도우 길이
#     #     hop_length (int): 스펙트로그램 계산에 사용되는 프레임 간격
#     #     window (str): 윈도우 함수의 종류
    
#     # Returns:
#     #     dict: 키(key)와 해당 키에 대응하는 피처 데이터로 구성된 딕셔너리

    
    
#     key2data = {}

#     with open(scp_path, 'r') as f:
#         for line in f.readlines():
#             #'0_01_0 E:\data\audio_mnist\data\01\0_01_0.wav'
#             key, path = line.split()

#             sig, rate = librosa.load(path, sr = sr)

#             feat = librosa.feature.mfcc(y = sig, sr=sr, n_mfcc=n_mfcc) # , n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window

#             key2data[key] = feat

#     return key2data
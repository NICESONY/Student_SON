import glob
import os
import numpy as np
import librosa
from libs.utils import load_dataset, set_length,preprocess_dataset, load_mfccs_from_directory





train_scp = 'E:/data/audio_mnist/scp/wav_audio_mnist_train.scp' # 경로 이 부분 배웠다. 다시 복습
dev_scp = 'E:/data/audio_mnist/scp/wav_audio_mnist_dev.scp' # 경로 이 부분 배웠다. 다시 복습
eval_scp = 'E:/data/audio_mnist/scp/wav_audio_mnist_eval.scp'# 경로 이 부분 배웠다. 다시 복습



def mfcc40_make(scp_path):


    dataset = []
    with open(scp_path, "r") as f:
        for line in f.readlines():
            key, input_path = line.strip().split()
            
            args = input_path.split('\\')
            args[4] = 'mfcc40\\eval'
            args[-1] = args[-1].replace('.wav', '')

            output_path = '\\'.join(args)  # E:\data\audio_mnist\mfcc40\01\0_01_15
            
            output_dir = '\\'.join(args[:-1])
            dataset.append(output_dir)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)


            # wav,  sr = librosa.load(input_path, sr=16000)


            # train_wav, train_ids = load_dataset(train_scp)
            # print(train_wav)


            eval_wav, eval_ids = load_dataset(eval_scp)

            # dev_wav, dev_ids = load_dataset(dev_scp)


            # train_wav = set_length(train_wav, 21000)
            # dev_wav = set_length(dev_wav, 21000)
            eval_wav = set_length(eval_wav, 21000)


            # print('train :', train_wav.shape)
            # print('dev :', dev_wav.shape)
            print('eval :', eval_wav.shape)



            mfccs = preprocess_dataset(eval_wav)


            # mfccs = librosa.feature.mfcc(y= wav, sr = sr,
            #                             n_mfcc = 40, 
            #                             hop_length = 160, 
            #                             win_length = 400,
            #                             n_fft = 512, 
            #                             window='hamming')
            print(mfccs.shape)
            
            

            # mfccs_resized = np.zeros((40, 132))
            # mfccs_resized[:, :mfccs.shape[1]] = mfccs[:, :132]
            np.save(output_path,  mfccs)
            print(mfccs.shape)

    
    print(dataset)
    return dataset
    

if __name__ == '__main__':
    mfcc40_make(eval_scp)
    
































##=================================================================




# import numpy as np
# import librosa
# import os

# # wav_files = 
# # output_dir = 

# trains_scp_file = 'E:/data/audio_mnist/wav_audio_mnist_train.scp'
# devs_scp_file = 'E:/data/audio_mnist/wav_audio_mnist_dev.scp'
# evals_scp_file =   'E:/data/audio_mnist/wav_audio_mnist_eval.scp'
# # data_dir = "E:/data/audio_mnist/data"  # WAV 파일들이 위치한 기본 디렉토리 경로
# output_dir = 'E:/data/audio_mnist/mfcc40'
# # 함수: 음성 파일을 MFCC로 변환하여 저장
# def save_mfccs(trains_scp_file, output_dir):
#     for wav_file in trains_scp_file:
#         # 파일 경로 및 이름 추출
#         file_name = os.path.splitext(os.path.basename(wav_file))[0]
#         output_file = os.path.join(output_dir, f"{file_name}.npy")
        
#         # MFCC 추출
#         wav, sr = librosa.load(wav_file)
#         mfccs = librosa.feature.mfcc(wav, sr=sr)
        
#         # MFCC 저장
#         np.save(output_file, mfccs)

# #     import os

# scp_file = "path/to/scp/file.scp"  # scp 파일 경로
# data_dir = "E:/data/audio_mnist/data"  # WAV 파일들이 위치한 기본 디렉토리 경로

    # # scp 파일 읽기
    # with open(trains_scp_file, 'r') as f:
    #     lines = f.readlines()

    # # WAV 파일 리스트 생성
    # train_wav_files = []
    # for line in lines:
    #     line = line.strip()  # 공백 및 개행 문자 제거
    #     parts = line.split(' ')  # 공백을 기준으로 분리
    #     wav_file = os.path.join(data_dir, parts[1])
    #     train_wav_files.append(wav_file)

    # print(train_wav_files)  # 훈련 데이터셋의 WAV 파일 경로 리스트 출력


#     # MFCC를 저장할 디렉토리 생성
#     os.makedirs("preprocessed_mfccs", exist_ok=True)

#     # 훈련 데이터셋의 MFCC 저장
#     train_wav_files = [...]  # 훈련 데이터셋의 WAV 파일 리스트
#     train_output_dir = "preprocessed_mfccs/train"
#     save_mfccs(train_wav_files, train_output_dir)

#     # 개발 데이터셋의 MFCC 저장
#     dev_wav_files = [...]  # 개발 데이터셋의 WAV 파일 리스트
#     dev_output_dir = "preprocessed_mfccs/dev"
#     save_mfccs(dev_wav_files, dev_output_dir)

#     # 평가 데이터셋의 MFCC 저장
#     eval_wav_files = [...]  # 평가 데이터셋의 WAV 파일 리스트
#     eval_output_dir = "preprocessed_mfccs/eval"
#     save_mfccs(eval_wav_files, eval_output_dir)


# if __name__ == "__main__" :
#     save_mfccs(trains_scp_file, output_dir)

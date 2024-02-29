import os
import numpy as np
import librosa
from multiprocessing import Process


# input
train_scp = 'E:/data/audio_mnist/scp/wav_audio_mnist_verification_train.scp' # 경로 이 부분 배웠다. 다시 복습
dev_scp = 'E:/data/audio_mnist/scp/wav_audio_mnist_verification_dev.scp' # 경로 이 부분 배웠다. 다시 복습
eval_scp = 'E:/data/audio_mnist/scp/wav_audio_mnist_verification_eval.scp'# 경로 이 부분 배웠다. 다시 복습

# output
mfcc_scp_format = 'E:/data/audio_mnist/scp/verification_mfcc{}_{}.scp'


def make_mfcc(scp_path, db_type, feat_dim: int = 40):


    dataset = []
    with open(scp_path, "r") as f_in, open(mfcc_scp_format.format(feat_dim, db_type), 'w') as f_out:
        for line in f_in.readlines():
            key, input_path = line.strip().split()
            
            args = input_path.split('\\')
            args[3] = f'verification_mfcc{feat_dim}'
            #args[4] = f'{db_type}'
            args[-1] = args[-1].replace('.wav', '')

            output_path = '\\'.join(args)  # E:\data\audio_mnist\verification_mfcc40\01\0_01_15
            
            output_dir = '\\'.join(args[:-1])
            dataset.append(output_dir)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            wav, sr = librosa.load(input_path, sr = 16000)


            mfccs = librosa.feature.mfcc(y= wav, sr = sr,
                                        n_mfcc = feat_dim, 
                                        hop_length = 160, 
                                        win_length = 400,
                                        n_fft = 512, 
                                        window='hamming')
            print(mfccs.shape)
            
            


            np.save(output_path,  mfccs)
            print(mfccs.shape)

            f_out.write(f'{key} {output_path}.npy\n')



    
    #print(dataset)

    

if __name__ == '__main__':
    #make_mfcc(train_scp, 'train')
    #make_mfcc(dev_scp, 'dev')
    #make_mfcc(eval_scp, 'eval')

    

    process_list = []
    """
    for wav_scp, db_key in [
        (train_scp, 'train'), 
        (dev_scp, 'dev'), 
        (eval_scp, 'eval')
        ]:
        proc = Process(target=make_mfcc, args=(wav_scp, db_key))
        process_list.append(proc)
    """

    proc = Process(target=make_mfcc, args=(train_scp, 'train'))
    process_list.append(proc)
    
    proc = Process(target=make_mfcc, args=(dev_scp, 'dev'))
    process_list.append(proc)

    proc = Process(target=make_mfcc, args=(eval_scp, 'eval'))
    process_list.append(proc)

    for p in process_list:
        p.start()
    for p in process_list:
        p.join()






























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

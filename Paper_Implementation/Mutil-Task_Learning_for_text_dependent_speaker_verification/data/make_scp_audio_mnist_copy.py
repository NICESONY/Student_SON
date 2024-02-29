# import glob
# from os.path import abspath, basename, splitext


# filename
# {class_id: 0~9}_{speaker_id: 01~60}_{session: 0~49}.wav

# path
# ./data/{01~60}/{filename}.wav

# Train: session 0~14 (15)
# Development: session 15~19 (5)
# Evaluation: session 20~49 (30)

import glob
from os.path import abspath, basename, splitext 

# glob 모듈은 파일 경로 패턴 매칭을 지원하며, 디렉토리 내 파일들의 목록을 검색하고 필터링하는 데 사용됩니다


wav_scp_train = "./wav_audio_mnist_train_note.scp"
wav_scp_dev = "./wav_audio_mnist_dev_note.scp"
wav_scp_eval = "./wav_audio_mnist_eval_note.scp"

def main():
    with open(wav_scp_train, "w") as f_train, open(wav_scp_dev, "w") as f_dev , open(wav_scp_eval, "w")as f_eval :
        for path in glob.glob("./data/??/?_??_*.wav"):
            path = abspath(path)

            key =  splitext(basename(path))[0]


            txt_id , spk_id, sess_id = key.split("_")

            print(key, txt_id, spk_id, sess_id)



# basename(path): 이 함수는 파일 경로(path)에서 파일의 기본 이름(파일 이름만)을 추출합니다. 
# 예를 들어, 만약 path가 "/경로/파일명.txt"와 같이 주어진다면, 이 함수는 "파일명.txt"를 반환합니다.

# splitext(): 이 함수는 파일 이름과 확장자를 분리합니다. 예를 들어, "파일명.txt"가 주어졌을 때, 
# 이 함수는 ("파일명", ".txt")와
# 같이 튜플로 결과를 반환합니다. 확장자를 제외한 파일 이름과 확장자를 분리하는 데 사용됩니다.


            sess_id = int(sess_id)


            if sess_id < 15 :
                f_train.write(f"{key} {path}\n")
            elif sess_id < 20 :
                f_dev.write(f"{key} {path}\n")
            else :
                f_eval.write(f"{key} {path}\n")
                
if __name__ == "__main__" :
    main()







# =========================================================================================

# wav_scp_train = './wav_audio_mnist_train.scp'
# wav_scp_dev = './wav_audio_mnist_dev.scp'
# wav_scp_eval = './wav_audio_mnist_eval.scp'


# def main():
#     with open(wav_scp_train, 'w') as f_train, open(wav_scp_dev, 'w') as f_dev, open(wav_scp_eval, 'w') as f_eval:
#         for path in glob.glob('./data/??/?_??_*.wav'):
#             path = abspath(path)

#             #args = path.split('\\')
#             #print(args, path)
#             #key = splitext(args[-1])[0]


#             # {class_id: 0~9}_{speaker_id: 01~60}_{session: 0~49}
#             key = splitext(basename(path))[0]

#             txt_id, spk_id, sess_id = key.split('_')

#             print(key, txt_id, spk_id, sess_id)  # 이것은 어디에 사용하는걸까?
#             # 이거는 cmd에 떠있다..



            #print(key, path)

        
            # sess_id = int(sess_id)

            # if sess_id < 15:
            #     # train
            #     f_train.write(f'{key} {path}\n') # 훈련용
            # elif sess_id < 20:
            #     # dev
            #     f_dev.write(f'{key} {path}\n') # dev는 무슨 용도일까?
            # else:
            #     # eval
            #     f_eval.write(f'{key} {path}\n') #평가용









# if __name__ == '__main__':
#     main()
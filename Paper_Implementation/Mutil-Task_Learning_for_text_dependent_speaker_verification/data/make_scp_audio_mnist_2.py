import glob
from os.path import abspath, basename, splitext
import os


# filename
# {class_id: 0~9}_{speaker_id: 01~60}_{session: 0~49}.wav

# path
# ./data/{01~60}/{filename}.wav

# Train: session 0~14 (15)
# Development: session 15~19 (5)
# Evaluation: session 20~49 (30)



scp_train = './wav_audio_mnist_train_2.scp'
scp_dev = './wav_audio_mnist_dev_2.scp'
scp_eval = './wav_audio_mnist_eval_2.scp'



# import json

# def main():
#     file_path = "E:/data/audio_mnist/data/audioMNIST_meta.txt"  # 실제 파일 경로로 변경해주세요
#     wav_scp_train = "train.scp"
#     wav_scp_dev = "dev.scp"
#     wav_scp_eval = "eval.scp"

#     # 파일 읽기
#     with open(file_path, 'r') as file:
#         data = json.load(file)

#     # 성별로 구분하기
#     male_data = {}
#     female_data = {}

#     for key, value in data.items():
#         gender = value["gender"]
#         if gender == "male":
#             male_data[key] = value
#         elif gender == "female":
#             female_data[key] = value

            
    
#     with open(wav_scp_train, 'w') as scp_train, open(wav_scp_dev, 'w') as scp_dev, open(wav_scp_eval, 'w') as scp_eval:
#         for path in glob.glob('./data/??/?_??_*.wav'):
#             path = abspath(path)

#             #args = path.split('\\')
#             #print(args, path)
#             #key = splitext(args[-1])[0]


#             # {class_id: 0~9}_{speaker_id: 01~60}_{session: 0~49}
#             key = splitext(basename(path))[0]

#             txt_id, spk_id, sess_id = key.split('_')

#             print(key, txt_id, spk_id, sess_id)

#         # 파일에 쓰기
#     # with open(wav_scp_train, 'w') as scp_train, open(wav_scp_dev, 'w') as scp_dev, open(wav_scp_eval, 'w') as scp_eval:
#     #     for key, value in male_data.items():
#     #         keys = int(key)
#     #         if keys < 15:
#     #             # train
#     #             scp_train.write(f'{key} {value}\n') # 훈련용
#     #         elif keys < 20:
#     #             # dev
#     #             scp_dev.write(f'{key} {value}\n') # dev는 무슨 용도일까?
#     #         else:
#     #             # eval
#     #             scp_eval.write(f'{key} {value}\n') #평가용

#     # 결과 출력
#     print("Male Data:")
#     for keys, value in male_data.items():
#         print(f"{keys}, {value['gender']}")

#     print("\nFemale Data:")
#     for keys, value in female_data.items():
#         print(f"{keys}, {value['gender']}")
    
    


#     for i in range(int(spk_id)) :
#         if i == int(keys) :
#             i, gender = value['gender']
#         print(keys)





# if __name__ == '__main__':
#     main()



scp_train = './wav_audio_mnist_train_2.scp'
scp_dev = './wav_audio_mnist_dev_2.scp'
scp_eval = './wav_audio_mnist_eval_2.scp'
file_female = "E:/data/audio_mnist/female.scp"
file_male = "E:/data/audio_mnist/male.scp"

def main():
    with open(scp_train, 'w') as f_train, open(scp_dev, 'w') as f_dev, open(scp_eval, 'w') as f_eval:
        for path in glob.glob('./data/??/?_??_*.wav'):
            path = abspath(path)

            # {class_id: 0~9}_{speaker_id: 01~60}_{session: 0~49}
            key = splitext(basename(path))[0]
            txt_id, spk_id, sess_id = key.split('_')

            print(key, txt_id, spk_id, sess_id)


            sess_id = int(sess_id)

            if sess_id < 15:
                # train
                f_train.write(f'{key} {path}\n') # 훈련용
            elif sess_id < 20:
                # dev
                f_dev.write(f'{key} {path}\n') # dev는 무슨 용도일까?
            else:
                # eval
                f_eval.write(f'{key} {path}\n') #평가용


    with open(file_male, 'r') as file:
        for line in file:
            key, gender = line.strip().split(', ')
            if gender == 'male':
                key = key.replace('??', '0')
            # 여기에 female에 대한 조건문을 추가하여 필요한 대체 작업을 수행할 수 있습니다.

            # 변경된 key를 사용하여 원하는 작업을 수행하면 됩니다.



if __name__ == '__main__':
    main()

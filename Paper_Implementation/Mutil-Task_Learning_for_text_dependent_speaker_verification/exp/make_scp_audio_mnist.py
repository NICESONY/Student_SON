import glob
from os.path import abspath, basename, splitext


# filename
# {class_id: 0~9}_{speaker_id: 01~60}_{session: 0~49}.wav

# path
# ./data/{01~60}/{filename}.wav

# Train: session 0~14 (15)
# Development: session 15~19 (5)
# Evaluation: session 20~49 (30)



wav_scp_train = './wav_audio_mnist_train.scp'
wav_scp_dev = './wav_audio_mnist_dev.scp'
wav_scp_eval = './wav_audio_mnist_eval.scp'


def main():
    with open(wav_scp_train, 'w') as f_train, open(wav_scp_dev, 'w') as f_dev, open(wav_scp_eval, 'w') as f_eval:
        for path in glob.glob('./data/??/?_??_*.wav'):
            path = abspath(path)

            #args = path.split('\\')
            #print(args, path)
            #key = splitext(args[-1])[0]


            # {class_id: 0~9}_{speaker_id: 01~60}_{session: 0~49}
            key = splitext(basename(path))[0]

            txt_id, spk_id, sess_id = key.split('_')

            print(key, txt_id, spk_id, sess_id)  # 이것은 어디에 사용하는걸까?
            # 이거는 cmd에 떠있다..





            #print(key, path)

        
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









if __name__ == '__main__':
    main()
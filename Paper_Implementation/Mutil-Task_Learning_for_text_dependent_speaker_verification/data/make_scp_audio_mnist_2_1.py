import glob
from os.path import abspath, basename, splitext
import os


# filename
# {class_id: 0~9}_{speaker_id: 01~60}_{session: 0~49}.wav

# path
# ./data/{01~60}/{filename}.wav

# Train: session 0~14 (15)
# Development: session 15~19 (5)
# trainuation: session 20~49 (30)




def main():
    input_file = "E:/data/audio_mnist/wav_audio_mnist_dev_2.scp"
    output_file = "E:/data/audio_mnist/wav_audio_mnist_dev_2_modified.scp"

    with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
        for line in input_f:
            key, path = line.strip().split(' ')
            value = key.split('_')[1]  # 01, 02, 03, ...
            
            if value in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                         '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                         '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                         '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                         '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                         '51', '52', '53', '54', '55']:
                # 값이 '01', '02', '03', '04'인 경우 0으로 변경
                key = key.replace(value, '0')
            elif value in ['55', '56', '57', '58', '59', '60']:
                # 값이 '55', '56', '57', '58', '59', '60'인 경우 1으로 변경
                key = key.replace(value, '1')

            # 변경된 key를 사용하여 원하는 작업을 수행하면 됩니다.
            output_f.write(f'{key} {path}\n')

if __name__ == '__main__':
    main()




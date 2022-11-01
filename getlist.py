import os

import librosa
import numpy as np
import pandas as pd
import wave

import soundfile
# 生成数据列表
def get_data_list(audio_path, list_path):
#获取训练集列表
    # sound_sum = 0#种类数量
    # audios = os.listdir(audio_path)
    #
    # f_train = open(os.path.join(list_path, 'train_list.csv'), 'w')
    #
    # for i in range(len(audios)):
    #     sounds = os.listdir(os.path.join(audio_path, audios[i]))
    #     for sound in sounds:
    #         if '.wav' not in sound:continue
    #         sound_path = os.path.join(audios[i], sound)
    #         f_train.write('\%s,%d\n' % (sound_path, i))
    #         sound_sum += 1
    #     print("Audio：%d/%d" % (i + 1, len(audios)))
    # f_train.close()
#获取测试集列表
    sound_sum = 0  # 种类数量
    # audios = os.listdir(audio_path)

    f_test = open(os.path.join(list_path, 'test_list.csv'), 'w')

    sounds = os.listdir(audio_path)
    for sound in sounds:
        if '.wav' not in sound: continue
        sound_path = os.path.join(audio_path, sound)
        f_test.write('\%s,%d\n' % (sound_path,31))
        sound_sum += 1
    # print("Audio：%d/%d" % (i + 1, len(audios)))
    f_test.close()

def get_submission_list(audio_path, list_path):
    sound_sum = 0  # 种类数量
    # audios = os.listdir(audio_path)

    f_test = open(os.path.join(list_path, 'submission.csv'), 'w')

    sounds = os.listdir(audio_path)
    for sound in sounds:
        if '.wav' not in sound: continue
        sound_path = os.path.join(audio_path, sound)
        f_test.write('%s\n' % (sound))
        sound_sum += 1
    print("共有音频：",sound_sum)
    f_test.close()

#测试用的，确认每段音频的时间，采样率，声道
def gettime(sound_path):
    max=0
    mid=0
    min=0
    sounds = os.listdir(sound_path)
    for sound in sounds:
        path = os.path.join(sound_path, sound)
        t = librosa.get_duration(filename=path)
        # d = wave.open(path).getnchannels()
        h = wave.open(path).getframerate()
        print(h)
        # if t>1:
        #     max=max+1
        # elif t==1:
        #     mid=mid+1
        # else:
        #     min=min+1
        # print("t=",t)
    print("max=",max)
    print("mid=",mid)
    print("min=",min)

if __name__ == '__main__':
    # get_data_list('train', '')
    # get_data_list('test', '')
    # get_submission_list('test', '')
    gettime("train/no")

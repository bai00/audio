import math, random

import numpy as np
import torch
import torchaudio
from torchaudio import transforms
# from IPython.display import Audio
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio


class AudioUtil():
  # ----------------------------
  # 加载音频文件，将信号作为张量返回，采样率返回
  # ----------------------------
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

  # ----------------------------
  # 统一声道
  # ----------------------------
  @staticmethod
  def rechannel(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
      # 如果声道满足条件，什么都不做
      return aud
    #转换为多声道
    if (new_channel == 1):
      # 通过仅选择第一个通道从立体声转换为单声道
      resig = sig[:1, :]
    else:
      # 通过复制第一个通道从单声道转换为立体声
      resig = torch.cat([sig, sig])

    return ((resig, sr))


  # ----------------------------
  # 统一采样率，由于采样率只适应单个声道，此时只对单个声道采样
  # ----------------------------
  @staticmethod
  def resample(aud, newsr):
    sig, sr = aud

    if (sr == newsr):
      #直接返回
      return aud

    num_channels = sig.shape[0]
    # 对第一个通道重新采样
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    #只有多声道的时候执行下列语句
    if (num_channels > 1):
      # 对第二个通道重新采样并合并两个通道
      retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
      resig = torch.cat([resig, retwo])

    return ((resig, newsr))

  # ----------------------------
  # 统一音频长度
  # ----------------------------
  @staticmethod
  def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr // 1000 * max_ms

    if (sig_len > max_len):
      # 将信号截断到指定长度
      sig = sig[:, :max_len]

    elif (sig_len < max_len):
      # 在信号开头和结尾填充长度
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len


      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)

    return (sig, sr)


  # ----------------------------
  # 将信号向左或向右移动一定百分比。末尾的值移到转换信号的开头。
  # ----------------------------
  @staticmethod
  def time_shift(aud, shift_limit):
    sig,sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)


  # ----------------------------
  # 生成梅尔声图谱
  # ----------------------------
  @staticmethod
  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)


  # ----------------------------
  #通过在两个频率中屏蔽频谱图的某些部分来增强频谱图
  # 屏蔽部分替换为平均值。
  # ----------------------------
  @staticmethod
  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec

# ----------------------------
# 声音数据集
# ----------------------------
class SoundDS(Dataset):
  def __init__(self, df, data_path):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 1000
    self.sr = 16000#44100
    self.channel = 1
    self.shift_pct = 0.4

  # ----------------------------
  # 数据集中的项数
  # ----------------------------
  def __len__(self):
    return len(self.df)

  # ----------------------------
  # 获取某项在数据集中的位置
  # ----------------------------
  def __getitem__(self, idx):
    # 音频文件的绝对文件路径，将音频目录与相对路径连接起来
    audio_file = self.data_path + self.df.loc[idx, 'relative_path']
    # 获取类别标签
    class_id = self.df.loc[idx, 'classID']

    aud = AudioUtil.open(audio_file)
    # 统一时间，采样率，声道以获取相同大小的数组
    reaud = AudioUtil.resample(aud, self.sr)
    rechan = AudioUtil.rechannel(reaud, self.channel)

    dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
    shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

    return aug_sgram, class_id

from torch.utils.data import random_split
import pandas as pd
from pathlib import Path

download_path = Path.cwd()

#训练数据集
train_list = download_path/'train_list.csv'
train_df = pd.read_csv(train_list)
train_df.head()
train_data_path = download_path/'train'
train_ds = SoundDS(train_df, train_data_path)
#-----------------------------------------------------------------------------------
#这一部分用于提交题目答案
#测试数据集
# test_list = download_path/'test_list.csv'
# test_df = pd.read_csv(test_list)
# test_df.head()
# test_data_path = download_path
# test_ds = SoundDS(test_df, test_data_path)


# train_ds, val_ds = train_ds,test_ds
# # Create training and validation data loaders
# train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
# val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
#这部分用于测试模型正确率，提交答案使用上面注释部分
num_items = len(train_ds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
#训练集，#测试集
train_ds, val_ds = random_split(train_ds, [num_train, num_val])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
#-----------------------------------------------------------------------------------

import torch.nn.functional as F
from torch.nn import init
from torch import nn

# ----------------------------
# 音频分类模型
# ----------------------------
class AudioClassifier(nn.Module):
  # ----------------------------
  # 构建模型体系结构
  # ----------------------------
  def __init__(self):
    super().__init__()
    conv_layers = []

    #第一个卷积快
    self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.relu1 = nn.ReLU()
    self.bn1 = nn.BatchNorm2d(8)
    init.kaiming_normal_(self.conv1.weight, a=0.1)
    self.conv1.bias.data.zero_()
    conv_layers += [self.conv1, self.relu1, self.bn1]


    #第二个卷积快
    self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    self.relu2 = nn.ReLU()
    self.bn2 = nn.BatchNorm2d(16)
    init.kaiming_normal_(self.conv2.weight, a=0.1)
    self.conv2.bias.data.zero_()
    conv_layers += [self.conv2, self.relu2, self.bn2]

    #第三个卷积快
    self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    self.relu3 = nn.ReLU()
    self.bn3 = nn.BatchNorm2d(32)
    init.kaiming_normal_(self.conv3.weight, a=0.1)
    self.conv3.bias.data.zero_()
    conv_layers += [self.conv3, self.relu3, self.bn3]

    #第四个卷积快
    self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    self.relu4 = nn.ReLU()
    self.bn4 = nn.BatchNorm2d(64)
    init.kaiming_normal_(self.conv4.weight, a=0.1)
    self.conv4.bias.data.zero_()
    conv_layers += [self.conv4, self.relu4, self.bn4]

    # 线性分类
    self.ap = nn.AdaptiveAvgPool2d(output_size=1)
    self.lin = nn.Linear(in_features=64, out_features=30)

    #包装卷积快
    self.conv = nn.Sequential(*conv_layers)

  # ----------------------------
  # 前向传递计算
  # ----------------------------
  def forward(self, x):
    # 运行卷积块
    x = self.conv(x)

    # 自适应扁平化，用于输入到线性层
    x = self.ap(x)
    x = x.view(x.shape[0], -1)

    # 线性层
    x = self.lin(x)

    # 输出
    return x


# 创建模型，有gpu用gpu，没有用cpu
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
#检查是不是cuda
next(myModel.parameters()).device


# ----------------------------
# 训练
# ----------------------------
def training(model, train_dl, num_epochs):
  # 损失函数、优化器和调度器
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                  steps_per_epoch=int(len(train_dl)),
                                                  epochs=num_epochs,
                                                  anneal_strategy='linear')

  # 对每次迭代重复此操作
  for epoch in range(num_epochs):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # 对训练集中的每个批次重复此操作
    for i, data in enumerate(train_dl):
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      # 标准化输入
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # 将参数梯度归零
      optimizer.zero_grad()


      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      scheduler.step()

      # 保留损失和准确性的统计数据
      running_loss += loss.item()

      # 获取得分最高的预测类
      _, prediction = torch.max(outputs, 1)
      # 与目标标签匹配的预测计数
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]

    # 打印统计信息
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction / total_prediction
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

  print('Finished Training')

num_epochs = 5
print('Start Training')
training(myModel, train_dl, num_epochs)


# ----------------------------
# 预测
# ----------------------------
def inference(model, val_dl):
  correct_prediction = 0
  total_prediction = 0
  label= torch.Tensor(0)

  # 禁用渐变更新
  with torch.no_grad():
    for data in val_dl:
      # 获取输入要素和目标标签
      inputs, labels = data[0].to(device), data[1].to(device)

      # 标准化输入
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # 获得预测
      outputs = model(inputs)

      # 获取得分最高的预测类
      _, prediction = torch.max(outputs, 1)
      label = torch.cat((label, prediction), axis=0)

      # 与目标标签匹配的预测计数
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
#以下注释部分也用于提交答案

  # label.tolist()
  # result = pd.read_csv("submission.csv")
  # result["label"]=label
  # result.to_csv("result.csv")

  acc = correct_prediction / total_prediction
  print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
  print("finished Inference")

# 执行预测
inference(myModel, val_dl)
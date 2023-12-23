from torch.utils.data.dataset import Dataset
import numpy
import soundfile
import os
import random
import torch

class FedDataset(Dataset):
    def __init__(self, train_list,train_path ,num_frames):
        self.train_path = train_path
        self.num_frames = num_frames
        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()

        dictkeys = list(set([x.split()[0] for x in lines]))  # dictkeys=labels
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)
    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(self.data_list[index])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)  # 转换为维度为一的张量
        return torch.FloatTensor(audio[0]), self.data_label[index]
    def __len__(self):
        return len(self.data_list)


class LocalDataset(Dataset):
    def __init__(self, train_list,train_path, num_frames):
        self.train_list = train_list
        self.num_frames = num_frames
        # Load data & labels
        self.data_list = []
        self.data_label = []
        # lines = open(train_list).read().splitlines()

        dictkeys = list(set([x.split()[0] for x in train_list]))  # dictkeys=labels
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        for index, line in enumerate(train_list):
            speaker_label = dictkeys[line.split()[0]]
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)
    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(self.data_list[index])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)  # 转换为维度为一的张量

        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)
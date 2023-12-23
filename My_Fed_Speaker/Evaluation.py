import torch
import copy

import torch.nn.functional as F
import  os
import sys
import argparse
import numpy
import tqdm
import soundfile
from Utils.Utils import *
from net.ecapa import ECAPA_TDNN
from Local_train import LocalClient
from Utils.tools import *

def eval_network(args,net, eval_list, eval_path):
    # net.to(args.device)
    net.eval()
    files = []
    embeddings = {}
    lines = open(eval_list).read().splitlines()
    for line in lines:
        files.append(line.split()[1])
        files.append(line.split()[2])
    setfiles = list(set(files))  # 去重利用集合去重
    setfiles.sort()

    for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):  # tqdm显示循环进度
        audio, _ = soundfile.read(os.path.join(eval_path, file))
        # Full utterance
        data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).to(args.device)

        # Spliited utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:  #
            shortage = max_audio - audio.shape[0]  # 如果帧长小于max_audio在后边补零
            audio = numpy.pad(audio, (0, shortage), 'wrap')  # （0.shortage代表在前边添加零个元素，后边添加shortage个元素）
        feats = []
        startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)  # 生成5个等间距的数值，每个帧长200有重叠
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])
        feats = numpy.stack(feats, axis=0).astype(numpy.float)  # 将feats列表堆叠成特征数组
        data_2 = torch.FloatTensor(feats).to(args.device)
        # Speaker embeddings
        with torch.no_grad():
            embedding_1 = net.forward(data_1, aug=False)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2 = net.forward(data_2, aug=False)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
        embeddings[file] = [embedding_1, embedding_2]  # embedding字典key=文件名,values=embedding未分帧和分帧
    scores, labels = [], []

    for line in lines:
        embedding_11, embedding_12 = embeddings[line.split()[1]]
        embedding_21, embedding_22 = embeddings[line.split()[2]]
        # Compute the scores
        score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
        score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))  # 计算矩阵相似度matmul矩阵相乘
        score = (score_1 + score_2) / 2
        score = score.detach().cpu().numpy()
        scores.append(score)
        labels.append(int(line.split()[0]))

    # Coumpute EER and minDCF
    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    return EER, minDCF

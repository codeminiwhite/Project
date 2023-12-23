
import os
import torch
import copy
import sys
import random
import argparse
import numpy as np
import time
data_list="/home/mengy23/data/voxceleb/train_list_1.txt"
class non_iid_datasplit():
    def __init__(self,data_list,clent_nums):
        self.data_list=data_list
        self.clent_nums=clent_nums
        self.lines = open(self.data_list).read().splitlines()
    def data_divide(self):
        print(type(self.lines))
        length=len(self.lines)
        clent_data_list=[]
        data_num=length//self.clent_nums
        for i in range(self.clent_nums):
            index=i*data_num
            clent_data_list=self.split_list(self.lines,data_num)
        clent_data_dict={ key:sub_list for key,sub_list in enumerate(clent_data_list)}
        return clent_data_list,clent_data_dict

    def split_list(self,lst, size):
        return [lst[i:i+size] for i in range(0, len(lst), size)]

class iid_datasplit():
    def __init__(self, data_list,clent_nums):
        self.data_list = data_list
        self.clent_nums=clent_nums
        self.lines = open(data_list).read().splitlines()

    def data_divide(self):

        print(type(self.lines))
        length = len(self.lines)
        clent_nums=self.clent_nums
        clent_data_list = []
        data_num = length // clent_nums
        data_idx_list=[i for i in range(length)]
        for i in range(clent_nums):
            subdata_list=[]
            random_items = set(sorted(random.sample(data_idx_list, data_num)))
            for i in random_items:
                subdata_list.append(self.lines[i])
            clent_data_list.append(subdata_list)
            data_idx_list=list(set(data_idx_list)-random_items)
        clent_data_dict = {key: sub_list for key, sub_list in enumerate(clent_data_list)}
        return clent_data_list, clent_data_dict




def Avg_weights(weight_list:list,frac_list:list):
    length=sum(frac_list)
    fracs=[]
    for i in frac_list:
        fracs.append(i/length)
    w_avg =copy.deepcopy(weight_list[0])
    for k in w_avg.keys():
        w_avg[k]=w_avg[k]*fracs[0]
        for i in range(1, len(weight_list)):
            w_avg[k] +=weight_list[i][k]*fracs[i]
    return w_avg


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))#进行除法操作
    return w_avg

def logger(log,save_path):
    log=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" "+str(log)
    with open(save_path+"/log.txt","a") as f:
        f.write(log+"\n")

# if __name__=="__main__":
#     start_time=time.time()
#     iid=iid_datasplit(data_list,120)
#     iid.data_divide()
#     data_list1,data_dict= iid.data_divide()
#     data_ls=data_dict[0]
#     =[1,2,3,4,7]ls
#     weigth=[{"key11":0.001,"key12":0.002,"key13":0.02,"key14":0.001},
#             {"key11":0.001,"key12":0.002,"key13":0.02,"key14":0.001},
#             {"key11":0.001,"key12":0.002,"key13":0.02,"key14":0.001},
#             {"key11":0.001,"key12":0.002,"key13":0.02,"key14":0.001},
#             {"key11":0.0009,"key12":0.002,"key13":0.04,"key14":0.001}]
#     avg=Avg_weights(weigth,ls)
#
#
#     # print(d)
#     # avg=FedAvg(weigth)
#     end_time = time.time()
#     total = end_time - start_time
#     print(0)



import copy

import torch
import torch.nn.functional as F
import  os
import sys
import argparse
import numpy
import tqdm
import soundfile
import time
from Utils.Utils import *
from net.ecapa import ECAPA_TDNN
from Local_train import LocalClient
from Utils.tools import *
from Evaluation import eval_network
vox1_dev_path = '/mnt/database/sre/voxceleb/1/dev/wav/'
vox2_dev_path = '/mnt/database/sre/voxceleb/vox2_wav/dev/aac/'


parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--clent_nums',  type=int,   default="50", help='total client nums')
parser.add_argument('--frac',  type=float,   default=0.02, help='total client nums')
parser.add_argument('--round',  type=int,   default=100,      help='Maximum number of epochs')
parser.add_argument('--n_cpu',      type=int,   default=16,       help='Number of loader threads')#DataLoader加载数据集用CPU加载
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/home/mengy23/data/voxceleb/train_list_1.txt", help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--trainpath', type=str,   default=vox1_dev_path, help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--eval_list',  type=str,   default="/home/mengy23/data/voxceleb/veri_test2.txt", help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="/mnt/database/sre/voxceleb/1/test/wav", help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')

parser.add_argument('--save_step',  type=int,   default=5, help='Path to save the score.txt and models')
parser.add_argument('--save_path',  type=str,   default="exps/vox2_aug", help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="", help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')
parser.add_argument('--embedding_size', type=int,   default=192,   help='Number of speakers')

parser.add_argument('--split',  type=str,   default="iid", help='data split type')
parser.add_argument("--local_epoch",type=int,default=1)
parser.add_argument('--numframes', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument("--local_batchsize",type=int,default=128)
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')
parser.add_argument("--loss",type=str,default="AAMsoftmax")
## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')
parser.add_argument('--gpu',  type=int,   default="0")
parser.add_argument('--seed',  type=int,   default="1")

## Initialization
torch.multiprocessing.set_sharing_strategy('file_system')#pytorch共享内存，将数据写入文件共享访问
args = parser.parse_args()
args = init_args(args)
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
#划分数据集
if args.split=='iid':
    data_split=iid_datasplit(data_list=args.train_list,clent_nums=args.clent_nums)
if args.split == 'non_iid':
    data_split = iid_datasplit(data_list=args.train_list, clent_nums=args.clent_nums)
else:
    data_split=iid_datasplit(data_list=args.train_list,clent_nums=args.clent_nums)
data_list,data_dict=data_split.data_divide()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

net_global=ECAPA_TDNN(C=args.C).to(args.device)
net_global.train()
w_global=net_global.state_dict()

if args.frac ==1:
    w_locals=[w_global for i in range(args.clent_nums)]
if args.frac <=0 or args.frac>1:
    print("error clients frac!!!")
m=max(int(args.frac * args.clent_nums),1)
for round in range(1,args.round+1):
    if args.frac!=1:
        w_locals=[]
    idxs_users = np.random.choice(range(args.clent_nums), m, replace=False)
    clents_acc=[]
    clents_loss=[]
    clent_data_length=[]
    for i in idxs_users:
        local=LocalClient(args,data_dict[i],w_glob=copy.deepcopy(w_global))
        w_local,loss,acc=local.train_network(round,i)
        if args.frac==1:
            w_locals[i]=copy.deepcopy(w_local)
            clent_data_length[i]=len(data_dict[i])
        else:
            w_locals.append(copy.deepcopy(w_local))
            clent_data_length.append(len(data_dict[i]))
        clents_loss.append(copy.deepcopy(loss))
        clents_acc.append(copy.deepcopy(acc))
    w_global=Avg_weights(w_locals,clent_data_length)
    net_global.load_state_dict(w_global)
    loss_avg = sum(clents_loss) / len(clents_loss)
    acc_avg=sum(clents_acc)/len(clents_acc)
    print('Round {:3d}, Average loss {:.5f}'.format(round, loss_avg))
    logger('Round {:3d}, Average loss {:.5f},Average acc{:2.2f}%'.format(round, loss_avg,acc_avg),args.save_path)
    if round % args.save_step==0:
        torch.save(net_global.state_dict(),args.save_path+"/{}_{}model_{:04d}.ckpt".format(args.round,args.clent_nums,round))
    if round % args.test_step==0:
        logger("Testing...",args.save_path)
        eer,minDCF=eval_network(args,net_global.to(args.device),args.eval_list,args.eval_path)
        logger("Testing end round: {:2d} clentnums:{:3d} eer: {:.5f}Testing minDCF:{:.5f}".format(args.round,args.clent_nums,eer,minDCF),args.save_path)
        print("Testing eer: {:.5f}".format(eer),"Testing minDCF:{:.5f}".format(minDCF))




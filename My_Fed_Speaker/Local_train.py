import sys
import os
import tqdm
import torch.nn as nn
import torch
import time
from loss import AAMsoftmax,AMSoftmax,Softmax
from net.ecapa import ECAPA_TDNN
from Utils.Dataset import LocalDataset
from  torch.utils.data.dataloader import DataLoader
class LocalClient(object):
	def __init__(self,args,trainlist,w_glob, **kwargs):
		## ECAPA-TDNN
		self.args=args
		self.dataset=LocalDataset(train_list=trainlist,train_path=args.trainpath,num_frames=args.numframes)
		self.loader=DataLoader(self.dataset,batch_size=args.local_batchsize,num_workers=16,shuffle=True)
		self.net= ECAPA_TDNN(C = self.args.C).to(args.device)
		self.net.load_state_dict(w_glob)
		# self.w=w_glob
		if (self.args.loss=="Softmax"):
			self.speaker_loss = Softmax.Softmax(embedding_size=self.args.embedding_size,n_class=self.args.n_class).to(args.device)
		if (self.args.loss=="AMsoftmax"):
			self.speaker_loss =AMSoftmax.AMSoftmax(embedding_size=self.args.embedding_size,n_class=self.args.n_class, m=self.args.m, s=self.args.s).to(args.device)
		if (self.args.loss=="AAMsoftmax"):
			self.speaker_loss = AAMsoftmax.AAMsoftmax(embedding_size=self.args.embedding_size,n_class=self.args.n_class, m=self.args.m, s=self.args.s).to(args.device)

		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=2e-5)  # 自适应优化器
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=self.args.test_step, gamma=self.args.lr_decay)
		## Classifier

		#用于调整优化器的学习率
		# print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.net.parameters()) / 1024 / 1024))

	def train_network(self, epoch,client):
		self.net.train()
		self.scheduler.step(epoch - 1)
		lr = self.optim.param_groups[0]['lr']

		epoch_loss=[]
		epoch_acc=[]
		for e in range(self.args.local_epoch):
			batch_loss=[]
			progbar=tqdm.tqdm(enumerate(self.loader,start=1),total=len(self.loader),ncols=100)
			index, top1, loss = 0, 0, 0
			# data:.wav files,labels:重新排列的batchsize个
			for num, (data, labels) in progbar:
				self.net.zero_grad()
				labels            = torch.LongTensor(labels).to(self.args.device)#与输入列表形状相同
				speaker_embedding =self.net.forward(data.to(self.args.device), aug = True)#提取192维的enmbedding
				nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)
				nloss.backward()
				self.optim.step()

				index += len(labels)
				top1 += prec.item()
				batch_loss.append(nloss.item())
				loss += nloss.detach().cpu().numpy()
				progbar.update()
				progbar.set_postfix(
					round=epoch,
					client=client,
					lr='{:.5f}'.format(lr),
					loss='{:.5f}'.format(loss/(num)),
					Acc='{:2.2f}%'.format(top1/index*len(labels))
				)
			progbar.close()
			epoch_loss.append(loss/num)
			epoch_acc.append(top1/index*len(labels))

		return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
			# 	sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			# 	" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			# 	" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			# 	sys.stderr.flush()
			# sys.stdout.write("\n")
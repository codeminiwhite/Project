
import torch
import torch.nn as nn


class Softmax(nn.Module):
    def __init__(self, embedding_size, n_class):
        super(Softmax, self).__init__()
        self.fc = nn.Linear(embedding_size, n_class)
        self.criertion = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        x = self.fc(x)
        loss = self.criertion(x, label.to(torch.long))
        pred = torch.softmax(x, -1)
        #有问题
        prec1= accuracy(pred.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1

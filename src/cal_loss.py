import torch
import torch.nn as nn

class cal_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(cal_loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, alpha):
        diff = input - target
        
        loss = torch.pow(diff, 2)  # 计算平方误差

        loss_1 = loss * (target < 1.2).float()    #计算小于1.2部分的loss
        loss_2 = loss * (target >= 1.2).float()    #计算大于1.2部分的loss

        loss = (1 - alpha) * loss_1 + alpha * loss_2   #将两部分进行加权融合，其中loss2权重大，loss1权重小
         
        if self.reduction == 'mean':
            return torch.mean(loss)  
        elif self.reduction == 'sum':
            return torch.sum(loss)  
        elif self.reduction == 'none':
            return loss  

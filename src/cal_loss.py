import torch
import torch.nn as nn

class cal_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(cal_loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        alpha = 7
        diff = input - target
        
        loss = torch.pow(diff, 2)  # 计算平方误差

        loss_1 = loss * (target < 1.2).float()    # 小于1.2的部分保持原样
        loss_2 = loss * (target >= 1.2).float()    # 大于1.2的部分loss翻倍为2

        loss = alpha * loss_1 + (1-alpha) * loss_2
        
        if self.reduction == 'mean':
            return torch.mean(loss)  
        elif self.reduction == 'sum':
            return torch.sum(loss)  
        elif self.reduction == 'none':
            return loss  

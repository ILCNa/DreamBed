from matplotlib import category
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np


class RSSM(nn.Module):
    # input_shape：表示输入数据的形状，通常是一个元组，包含输入数据的维度信息
    # hidden_size：用来存储历史信息的隐藏状态的维度大小
    # output_size：表示输出的维度大小，通常用于表示模型的预测结果或潜在表示的维度
    # fullstate = (h, z) 其中 h 是 GRU 的隐藏状态，z 是潜在状态,其维度是 hidden_size + output_size
    def __init__(self, input_shape, hidden_size, action_dim=3,output_size=16):
        super(RSSM, self).__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_size = output_size
    # 预测奖励
        self.reward_predictor = nn.Linear(hidden_size+output_size, 1)  # 预测奖励的线性层
    # 预测终止
        self.continue_predictor = nn.Linear(hidden_size+output_size, 1)  # 预测终止的线性层
    # 解码器神经网络
        self.dec = nn.Sequential(
            nn.Linear(output_size+hidden_size, 4608), # 16+16 -> 4608
            nn.SiLU(),
            nn.Unflatten(1, (128, 6, 6)), # 4608 -> 128*6*6
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1), # 128*6*6 -> 64*12*12
            nn.SiLU(),
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1), # 64*12*12 -> 32*24*24
            nn.SiLU(),
            nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1), # 32*24*24 -> 16*48*48
            nn.SiLU(),
            nn.ConvTranspose2d(16,3,kernel_size=4,stride=2,padding=1), # 16*48*48 -> 3*96*96
            nn.Sigmoid()
        )
    # 先验z
        self.prior_head = nn.Linear(hidden_size, output_size) 
    # 算后验Z
        self.posterior_head = nn.Linear(4608+hidden_size, output_size)
    # GRU： 
        self.gru = nn.GRUCell(input_size=output_size+action_dim, hidden_size=hidden_size)
    # 论文里的enc
        self.enc = nn.Sequential(
        nn.Conv2d(3,16,kernel_size=4,stride=2,padding=1), #96,96,3 -> 48,48,16
        nn.SiLU(),
        nn.Conv2d(16,32,kernel_size=4,stride=2,padding=1), # 48,48,16 -> 24,24,32
        nn.SiLU(),
        nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1), # 24,24,32 -> 12,12,64
        nn.SiLU(),
        nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),# 12,12,64 -> 6,6,128
        nn.SiLU(),
        nn.Flatten(start_dim=1),# 6,6,128 -> 4608
        )

    def reward_predict(self,h,z):
        """
    Reward predictor:
        用来预测奖励，输入为当前隐藏状态h和潜在状态z即传说中的全状态，输出为预测的奖励值。
        """
        x = torch.cat((h,z.flatten(start_dim=1)),dim=-1)
        reward = self.reward_predictor(x)
        return reward.squeeze(-1)  
    def continue_predict(self,h,z):
        """
    Continue predictor:
        用来预测终止，输入为当前隐藏状态h和潜在状态z，输出为预测的终止值。
        """
        x = torch.cat((h,z.flatten(start_dim=1)),dim=-1)
        x = self.continue_predictor(x)
       # x = torch.sigmoid(x) # 预测终止值在0-1之间
        #print(x,x.shape) [2,1]
        x = x.squeeze(-1) # 
       #print(x,x.shape) [2]
        return x

# 求后验Z之前的logitis

    def posterior_logits(self,x,h):
        """
Encoder:
            用来计算后验分布的logits，输入为当前观测x和隐藏状态h，输出为后验分布的logits，最后对其分组（因子化动力学预测器）变成Z
"""
        # 将输入x和隐藏状态h进行拼接
        x = self.enc(x)
        x = torch.cat((x,h),dim=-1)
        # 通过posterior_head计算后验分布的logits
        logits  = self.posterior_head(x)
        return logits.reshape(logits.shape[0],4,4)# 对应论文的RSSM中的Z，将h与x拼接后经过MLP再进行分组
    # 对Z分组是为了将其分为4个类别有4个可能的取值，factorized dynamics predictor for fast sampling during imagination training
    
# 求h——t+1
    def next_hidden(self,z,a,h):
        """
Sequence model:
        将z与a进行拼接，然后通过GRUCell计算下一个隐藏状态h，输入为当前的潜在状态z、动作a和当前隐藏状态h，输出为下一个隐藏状态h
        """
        z= z.flatten(start_dim=1)
        gru_input = torch.cat((z,a),dim=-1)
        return self.gru(gru_input,h)

# 先验Z
    def prior_logits(self,h):
        """
Dynamics predictor:
        用来计算先验分布的logits，输入为当前隐藏状态h，输出为先验分布的logits.
    ※需要进行训练，使其对应的先验分布与后验分布尽可能接近，从而使模型能够更好地预测未来的状态。
        """
        logits = self.prior_head(h)
        # B,4,4
        return logits.reshape(logits.shape[0],4,4) # 对应论文的RSSM中的Z，将h经过MLP再进行分组
    
    def imagine_step(self,z,a,h):
        """
        用来进行想象步骤，输入为当前的潜在状态z、动作a和当前隐藏状态h，输出为下一个潜在状态z'和下一个隐藏状态h'。
        """
        # 获得gru的nexth
    
        h_next = self.next_hidden(z,a,h)
        logits = self.prior_logits(h_next)
        # 4*4 
        categorical = torch.distributions.Categorical(logits=logits) # 
        indexes = categorical.sample()#1*4
        z_next = F.one_hot(indexes,num_classes=logits.shape[-1]).to(logits.dtype) # 1*4*4
        # 尽管是1*4，[0]【X]前的[0]也不能省略
        #print("四个选项的概率：", categorical.probs[0, 0].tolist())
        #print("抽中的编号：", indexes[0, 0].item())
      #  print("编号写成向量：", z_next[0, 0].tolist())
        return h_next, z_next
    def decode(self,h,z):
        """
Decode:
        用来解码潜在状态z，输入为潜在状态z，输出为解码后的观测x'。
        """
        # 这里可以使用一个简单的线性层作为解码器，也可以使用更复杂的网络结构
        # 例如卷积神经网络或反卷积神经网络
        # 这里为了简化，使用一个线性层作为解码器
        x = torch.cat((z.flatten(start_dim=1),h),dim=-1) # 16+
        x = self.dec(x)
        return x
# 正在的球后验Z    
    def sample_z(self,logits):
        categorical = torch.distributions.Categorical(logits=logits)
        indexes = categorical.sample()
        z_hard = F.one_hot(indexes,num_classes=logits.shape[-1]).to(logits.dtype)
        probs = categorical.probs
        #z_soft = probs
        return z_hard+(probs-probs.detach())# 由于 z_hard是one-hot向量，梯度为0
    #所以这里使用了一个技巧，将z_hard和z_soft相加，使得梯度可以传递到z_soft上，从而实现了对z的采样。
        
    def forward(self, x):
        h = torch.zeros(1, self.hidden_size)
        return x

if __name__ == "__main__":
    torch.manual_seed(0)
    model = RSSM(
        input_shape=(96, 96, 3),
        hidden_size=16,
        action_dim=3,
        output_size=16,
    )

    x0 = torch.randn(1, 3, 96, 96)       # 一张假图
    h0 = torch.zeros(1, 16)              # 初始记忆
    a0 = torch.tensor([[0.0, 0.5, 0.0]]) # 方向居中、半油门、不刹车
    h = torch.zeros(2, model.hidden_size)
    codes = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
    z = F.one_hot(codes, num_classes=4).float()

    reward = model.reward_predict(h, z)
    continue_logits = model.continue_predict(h, z)
    assert reward.shape == continue_logits.shape == (2,)

    model.zero_grad()
    loss_reward = F.mse_loss(reward, torch.tensor([0.5, -1.0]))
    loss_continue = F.binary_cross_entropy_with_logits(
        continue_logits, torch.tensor([1.0, 0.0])
    )
    (loss_reward + loss_continue).backward()

    assert model.reward_predictor.weight.grad.abs().sum().item() > 0
    assert model.continue_predictor.weight.grad.abs().sum().item() > 0
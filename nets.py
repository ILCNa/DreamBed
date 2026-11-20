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
    def __init__(self, input_shape, hidden_size, action_dim=3,output_size=16):
        super(RSSM, self).__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_size = output_size
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
        print("四个选项的概率：", categorical.probs[0, 0].tolist())
        print("抽中的编号：", indexes[0, 0].item())
        print("编号写成向量：", z_next[0, 0].tolist())
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

    with torch.no_grad():
        logits = model.posterior_logits(x0, h0)
        indices = torch.distributions.Categorical(logits=logits).sample()
        z0 = F.one_hot(indices, num_classes=4).to(logits.dtype)
        h1 = model.next_hidden(z0, a0, h0)
                # 还没看下一帧，只用 h1 从先验采样
        prior1 = model.prior_logits(h1)
        prior_indices = torch.distributions.Categorical(logits=prior1).sample()

        # 拿到下一帧后，用同一个 h1 算后验
        x1 = torch.randn_like(x0)
        post1 = model.posterior_logits(x1, h1)

        assert prior1.shape == post1.shape == (1, 4, 4)
        assert torch.isfinite(prior1).all()
        assert prior_indices.shape == (1, 4)

        print("先验和后验的形状：", tuple(prior1.shape), tuple(post1.shape))
        print("先验抽中的四个类别：", prior_indices.tolist())
        a1 = torch.tensor([[0.2, 0.3, 0.0]])  # 稍向右转，给一点油门

        h1_img, z1_img = model.imagine_step(z0, a0, h0)
        h2_img, z2_img = model.imagine_step(z1_img, a1, h1_img)

        assert h1_img.shape == h2_img.shape == (1, 16)
        assert z1_img.shape == z2_img.shape == (1, 4, 4)
        assert (z2_img.sum(dim=-1) == 1).all()

        print("想象第 1 步：", tuple(h1_img.shape), tuple(z1_img.shape))
        print("想象第 2 步：", tuple(h2_img.shape), tuple(z2_img.shape))
        x0_recon = model.decode(h0, z0)          # z0 来自看过 x0 的后验
        x2_pred = model.decode(h2_img, z2_img)    # z2_img 来自只给动作的先验



        print("重建第 0 帧：", tuple(x0_recon.shape))
        print("预测第 2 帧：", tuple(x2_pred.shape))
        model.zero_grad()
    x0 = torch.rand(1, 3, 96, 96)  # 用 [0, 1] 内的假像素
    h0 = torch.zeros(1, 16)

    logits = model.posterior_logits(x0, h0)
    z0 = model.sample_z(logits)
    x0_recon = model.decode(h0, z0)

    assert ((z0 == 0) | (z0 == 1)).all()
    assert (z0.sum(dim=-1) == 1).all()
    assert ((x0_recon >= 0) & (x0_recon <= 1)).all()

    loss = F.mse_loss(x0_recon, x0)
    loss.backward()

    grad = model.posterior_head.weight.grad
    assert grad is not None, "重建误差还没传到后验头"
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0

    print("重建误差：", loss.item())
    print("后验头梯度绝对值之和：", grad.abs().sum().item())
    group_sums = z0.sum(dim=-1)
    assert logits.shape == (1, 4, 4)
    assert z0.shape == (1, 4, 4)
    assert (group_sums == 1).all()
    assert h1.shape == (1, 16)
    assert torch.isfinite(h1).all()

    print("抽中的类别编号：", indices.tolist())
    print("z0 每组的和：", group_sums.tolist())
    print("h1 的形状：", tuple(h1.shape))
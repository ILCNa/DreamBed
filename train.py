import h11
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
from nets import RSSM

from tqdm import tqdm

seed = 42 # 宇宙的答案

recon_losses = []

zPp_KL = []#先验学后验
zpP_KL = []#后验学先验

def img_to_tensor(img):
    # permute： 更改张量的维度顺序，unsqueeze：在指定位置增加一个维度，/255.0：将像素值归一化到0-1之间
# 卷积网络 CHW
    return torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0



def reconstruction_train(rssm:RSSM,
                         frames:list,actions:list,rewards:list,terminated:list,truncated:list
                         ,h0=torch.zeros(1,16),train_steps=250):
    assert len(frames) == len(actions) + 1, "帧数应该比动作数多 1"
    assert len(actions) > 0, "至少需要一个动作来检查 GRU"
    
    optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-3)
    for _ in tqdm(range(train_steps)):
        print(f"第{_+1}步训练:")
        h = h0  
        losses = []
        dynamics_loss = []
        representations_loss = []
        rewards_loss = []
        continues_loss = []
        optimizer.zero_grad()
        for i in tqdm(range(len(frames)),desc="重建帧:"):
        
            logits = rssm.posterior_logits(frames[i], h)
            z = rssm.sample_z(logits)# 后验
            if i>= 1:
                # 先验向后验学习准确性。
                # KL 先获得俩个分类分布，再计算 KL 散度
                logits_p = rssm.prior_logits(h) # 先验
                catergorical_pre = torch.distributions.Categorical(logits=logits_p)
                catergorical_post = torch.distributions.Categorical(logits=logits.detach())# detach() 使后验不参与梯度计算
                kl_div = torch.distributions.kl_divergence(catergorical_post, catergorical_pre)
                dynamics_loss.append(kl_div)# [tensor(0.1), tensor(0.2), tensor(0.3),tensor(0.4)] 

                # 后验向先验学习后者的输出习惯，如此可以更快地收敛到一个稳定的状态。因为先验是基于历史信息的，而后验是基于当前观测的。通过让后验向先验学习，可以让模型更好地利用历史信息，从而提高预测的准确性。
                catergorical_pre = torch.distributions.Categorical(logits=logits_p.detach())# detach() 使先验不参与梯度计算
                catergorical_post = torch.distributions.Categorical(logits=logits)
                kl_div = torch.distributions.kl_divergence( catergorical_post,catergorical_pre)
                representations_loss.append(kl_div)

                r = rssm.reward_predict(h, z)
                c = rssm.continue_predict(h, z)
                r_target = rewards[i-1]
                c_target = 1.0 - terminated[i-1] - truncated[i-1] # 终止或截断时，c_target为0，否则为1

                rewards_loss.append(F.mse_loss(r, r_target))
                continues_loss.append(F.binary_cross_entropy_with_logits(c, c_target))



            recon = rssm.decode(h, z)
            losses.append(F.mse_loss(recon, frames[i]))# 这里用了loss数组，这意味着
            # 每一帧都要进行重建，不过重建过程中h是不断更新的，h是GRU的隐藏状态，随着每一帧的输入和动作的变化而变化


            if i < len(actions):
                h = rssm.next_hidden(z, actions[i], h)
            
        # 将 4 个 tensor 合而唯一
        """[tensor(0.2094, grad_fn=<MseLossBackward0>), tensor(0.2094, grad_fn=<MseLossBackward0>), tensor(0.2094, grad_fn=<MseLossBackward0>), tensor(0.2094, grad_fn=<MseLossBackward0>)]
    =>tensor([0.2094, 0.2094, 0.2094, 0.2094], grad_fn=<StackBackward0>)"""
        loss = torch.stack(losses)# 可以把多个形状相同的张量叠成一个张量。,dim决定新维度在哪,默认为0.就是多了行维度。
        loss = loss.mean()# 取平均值，得到一个标量
        recon_losses.append(loss.item())
        zPp_KL.append(torch.stack(dynamics_loss).mean().item())
        zpP_KL.append(torch.stack(representations_loss).mean().item())
        loss += torch.stack(rewards_loss).mean()
        loss +=torch.stack(continues_loss).mean()
        loss+= torch.stack(dynamics_loss).mean()  # 如果有动态损失，就加上第一个动态损失，否则加0
        loss+= torch.stack(representations_loss).mean()  # 如果有表示损失，就加上第一个表示损失，否则加0
        loss.backward()
        optimizer.step()


        if(_+1) % 1000 == 0:
            save(rssm, f"rssm_checkpoint.pt")

def plot_losses():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.title('Reconstruction Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(zPp_KL, label='先验学后验', color='orange')
    plt.title('KL Divergence Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(zpP_KL, label='后验学先验', color='orange')
    plt.title('KL Divergence Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    # totol_loss=  zip(recon_losses, zpp_KL)
    # plt.plot(totol_loss, label='Total Loss', color='green')
    plt.show()
def save(rssm:RSSM, path:str):
    torch.save(rssm.state_dict(), path)
    print(f"模型已保存到 {path}")
def load(rssm:RSSM, path:str):
    rssm.load_state_dict(torch.load(path))
    print(f"模型已从 {path} 加载")









def test_reconsturction(steps = 15):
    """
    测试重建效果。
    环境跑15帧，动作随机。
    让rssm学习预测15帧
    输出loss
    输出可视化结果，对照原15帧与预测的。
    """
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    obs,_ = env.reset(seed=seed)# 96 96 3 的图像
    obs = img_to_tensor(obs)
    rssm = RSSM(input_shape=(3,96,96), hidden_size=16, action_dim=3, output_size=16)
    frames_real = []
    frames_pred = []
    actions= []
    rewards = []
    terminateds = []
    truncateds = []
    frames_real.append(obs)
    for _ in range(steps):
        action = env.action_space.sample() # 随机采样动作
        
        obs,reward,terminated ,truncated,_ = env.step(action)
        frames_real.append(img_to_tensor(obs))
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0) # 转为张量
        actions.append(action)
        rewards.append(torch.tensor(reward, dtype=torch.float32).unsqueeze(0))
        terminateds.append(torch.tensor(terminated, dtype=torch.float32).unsqueeze(0))
        truncateds.append(torch.tensor(truncated, dtype=torch.float32).unsqueeze(0))


    reconstruction_train(rssm=rssm,frames=frames_real,actions=actions,rewards=rewards,terminated=terminateds,truncated=truncateds)
    predicted_signals = []
    with torch.no_grad():
        h = torch.zeros(1, rssm.hidden_size)
        z = rssm.sample_z(rssm.posterior_logits(frames_real[0], h))
        for action in actions:
            h, z = rssm.imagine_step(z, action, h)
            frames_pred.append(rssm.decode(h, z))
            # 三项预测都来自当前想象状态 (h, z)，因此与同一帧对齐。
            reward_pred = rssm.reward_predict(h, z).item()
            continue_prob = rssm.continue_predict(h, z).sigmoid().item()
            predicted_signals.append((reward_pred, continue_prob))

# 可视化结果：
    n = len(frames_pred)
    assert n > 0 and len(frames_real) == n + 1

    cols = min(5, n)
    pairs = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        2 * pairs, cols, figsize=(2.4 * cols, 4.8 * pairs),
        squeeze=False, layout="constrained",
    )

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for i, pred in enumerate(frames_pred):
        real = frames_real[i + 1]
        row, col = divmod(i, cols)
        mse = F.mse_loss(pred, real).item()
        print(f"第 {i + 1:2d} 步 MSE：{mse:.6f}")

        axes[2 * row, col].imshow(real[0].permute(1, 2, 0).detach().cpu().numpy())
        axes[2 * row, col].set_title(f"Real {i + 1}")

        pred_ax = axes[2 * row + 1, col]
        pred_ax.imshow(pred[0].permute(1, 2, 0).detach().cpu().numpy())
        pred_ax.set_title(f"Pred {i + 1} · MSE {mse:.4f}", fontsize=10)

        reward_pred, continue_prob = predicted_signals[i]
        pred_ax.set_xlabel(
            f"Reward {reward_pred:+.3f}  |  P(cont) {continue_prob:.1%}", fontsize=9,
        )

    plot_losses()


if __name__ == "__main__":
    test_reconsturction(steps=15)



# for _ in range(10):
#     action = env.action_space.sample() # 随机采样动作
#     obs, reward, terminated, truncated, info = env.step(action) # 执行动作, 返回 observation, reward, terminated, truncated, info
#     x1 = torch.tensor(obs, dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0
#     posterior_logits = rssm.posterior_logits(x1, h0)
#     z1 = rssm.sample_z(posterior_logits)
#     loss = F.mse_loss(x1, rssm.decode(h0, z1))
#     loss.backward()
#     after = loss.item()
#     torch.optim.Adam(rssm.parameters(), lr=1e-3).step()
import h11
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
from nets import RSSM

from tqdm import tqdm

seed = 42 # 宇宙的答案

def img_to_tensor(img):
    # permute： 更改张量的维度顺序，unsqueeze：在指定位置增加一个维度，/255.0：将像素值归一化到0-1之间
# 卷积网络 CHW
    return torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0



def reconstruction_train(rssm:RSSM,frames:list,actions:list,h0=torch.zeros(1,16),train_steps=2500):
    assert len(frames) == len(actions) + 1, "帧数应该比动作数多 1"
    assert len(actions) > 0, "至少需要一个动作来检查 GRU"
    
    optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-3)
    for _ in tqdm(range(train_steps)):
        print(f"第{_+1}步训练:")
        h = h0  
        losses = []
        optimizer.zero_grad()
        for i in tqdm(range(len(frames)),desc="重建帧:"):
        
            logits = rssm.posterior_logits(frames[i], h)
            z = rssm.sample_z(logits)

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
        loss.backward()
        optimizer.step()
        if(_+1) % 1000 == 0:
            save(rssm, f"rssm_checkpoint.pt")
    print(f"训练后：zloss:{loss.item():.6f}")
    print(f"更新后的h0:{h0}")
def save(rssm:RSSM, path:str):
    torch.save(rssm.state_dict(), path)
    print(f"模型已保存到 {path}")
def load(rssm:RSSM, path:str):
    rssm.load_state_dict(torch.load(path))
    print(f"模型已从 {path} 加载")
def test_reconsturction(steps = 15):
    """
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
    frames_real.append(obs)
    for _ in range(steps):
        action = env.action_space.sample() # 随机采样动作
        
        obs,reward,_,_,_ = env.step(action)
        frames_real.append(img_to_tensor(obs))
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0) # 转为张量
        actions.append(action)

    reconstruction_train(rssm=rssm,frames=frames_real,actions=actions)
    with torch.no_grad():
        h = torch.zeros(1, rssm.hidden_size)
        z = rssm.sample_z(rssm.posterior_logits(frames_real[0], h))
        for action in actions:
            h, z = rssm.imagine_step(z, action, h)
            frames_pred.append(rssm.decode(h, z))


    n = len(frames_pred)
    assert n > 0 and len(frames_real) == n + 1

    cols = min(5, n)
    pairs = (n + cols - 1) // cols
    fig, axes = plt.subplots(2 * pairs, cols, figsize=(2.4 * cols, 2.3 * pairs * 2),
                            squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for i, pred in enumerate(frames_pred):
        real = frames_real[i + 1]
        row, col = divmod(i, cols)
        mse = F.mse_loss(pred, real).item()
        print(f"第 {i + 1:2d} 步 MSE：{mse:.6f}")

        axes[2 * row, col].imshow(real[0].permute(1, 2, 0).detach().cpu().numpy())
        axes[2 * row, col].set_title(f"Real {i + 1}")

        axes[2 * row + 1, col].imshow(pred[0].permute(1, 2, 0).detach().cpu().numpy())
        axes[2 * row + 1, col].set_title(f"Pred {i + 1} · MSE {mse:.4f}")

    plt.tight_layout()
    plt.show()


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
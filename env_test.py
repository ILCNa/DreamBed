import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import gymnasium as gym

seed = 42 # 宇宙的答案

env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
def test_env(env, seed):
    frames = []
    transitions = []
    print("Testing environment with seed:", seed)
    print(env.reset(seed=seed))# 96 96 3 的图像
    print(env.observation_space) # Box(96, 96, 3)
    print(env.action_space) # Box(-1.0, 1.0, (3,), float32)
    for _ in range(3):
        action = env.action_space.sample() # 随机采样动作
        print("Sampled action:", action)
        obs, reward, terminated, truncated, info = env.step(action) # 执行动作, 返回 observation, reward, terminated, truncated, info
        print(obs, reward, terminated, truncated, info)
        frames.append(obs)
        transitions.append((obs, action, reward, terminated, truncated, info))
    return frames, transitions
    # 显示帧
    # for i, frame in enumerate(frames):
    #     plt.subplot(1,3 , i + 1)
    #     plt.imshow(frame)
    #     plt.title(f"observation {i + 1}")
    #     plt.axis("off")
    # plt.show()
#test_env(env, seed)

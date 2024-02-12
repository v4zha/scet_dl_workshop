import torch
import pybullet_envs
import numpy as np
import gym
from tqdm import tqdm
from scet_rl import TD3
from scet_rl import AgentMode

env = gym.make("Walker2DBulletEnv-v0", render=True)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action_val = env.action_space.high[0]
agent = TD3(state_dim=state_dim, action_dim=action_dim, max_action_val=max_action_val,
            checkpt_file_path="checkpoints/walker2dbullet_td3", load_checkpoint=True, actor_lr=1e-3, critic_lr=1e-3, log_dir="logs/walker2dbullet_td3_eval", mode=AgentMode.EVAL)
print("device : ", agent.device)
logger = agent.writer
chekpt_interval = 100
batch_size = 100
max_episodes = 100
warmup_steps = 0
initial_steps = 0
reward_history = []

for ep in tqdm(range(max_episodes+1)):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    ep_reward = 0
    done = False
    while not done:
        if initial_steps < warmup_steps:
            action = env.action_space.sample()
            action = torch.tensor(action, dtype=torch.float32)
            initial_steps += 1
        else:
            action = agent.get_action(state)
        action_np = action.cpu().numpy()
        next_state, reward, done, _info = env.step(action_np)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        state = next_state
        ep_reward += reward

    reward_history.append(ep_reward)
    avg_reward = np.mean(reward_history[-100:])

    print("\nEpisode : ", ep, " Episode Reward : %.1f" %
          ep_reward, " Avg Reward : %.1f" % avg_reward)
    logger.add_scalar("Episode Reward", ep_reward, ep)
    logger.add_scalar("Avg Reward", avg_reward, ep)

env.close()
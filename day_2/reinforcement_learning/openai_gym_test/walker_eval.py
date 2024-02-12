import gymnasium as gym
import torch
import numpy as np
from scet_rl import TD3
from tqdm import tqdm
from scet_rl import AgentMode

env = gym.make("BipedalWalker-v3", render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action_val = env.action_space.high[0]
model = TD3(state_dim=state_dim, action_dim=action_dim, max_action_val=max_action_val,
            checkpt_file_path="checkpoints/bipedal_walker_td3", load_checkpoint=True, actor_lr=3e-4, critic_lr=3e-4, mode=AgentMode.EVAL, log_dir="logs/bipedal_walker_td3_eval")
print("device : ", model.device)
logger = model.writer
batch_size = 100
max_episodes = 2000
init_steps = 0
reward_history = []
for episodes in range(max_episodes+1):
    ep_reward = 0
    state = env.reset()
    state = torch.tensor(state[0])
    actor_loss_ep = 0
    critic_loss_ep = 0
    train_count = 0
    done = False
    truncated = False
    while not done and not truncated:
        action = model.get_action(state)
        action_np = action.cpu().numpy()
        next_state, reward, done, truncated, _info = env.step(action_np)
        next_state = torch.tensor(next_state)
        state = next_state
        ep_reward += reward
    reward_history.append(ep_reward)
    avg_reward = np.mean(reward_history[-100:])
    print("\nEpisode : ", episodes, " Episode Reward : %.1f"
          % ep_reward, " Avg Reward : %.1f" % avg_reward)

env.close()

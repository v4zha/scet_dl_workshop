import gymnasium as gym
import torch
import numpy as np
from scet_rl import TD3
from tqdm import tqdm


resume_training = False
env = gym.make("BipedalWalker-v3")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action_val = env.action_space.high[0]
model = TD3(state_dim=state_dim, action_dim=action_dim, max_action_val=max_action_val,
            checkpt_file_path="checkpoints/bipedal_walker_td3", load_checkpoint=resume_training, actor_lr=1e-3, critic_lr=1e-3, log_dir="logs/bipedal_walker_td3_train")
print("device : ", model.device)
logger = model.writer
chekpt_interval = 100
batch_size = 100
max_episodes = 2000
warmup_steps = 1000 if not resume_training else 0
init_steps = 0
best_reward = env.reward_range[0]
reward_history = []
for episodes in tqdm(range(max_episodes+1)):
    ep_reward = 0
    state = env.reset()
    state = torch.tensor(state[0])
    actor_loss_ep = 0
    critic_loss_ep = 0
    train_count = 0
    done = False
    truncated = False
    while not done and not truncated:
        if init_steps < warmup_steps:
            action = env.action_space.sample()
            action = torch.tensor(action)
            init_steps += 1
        else:
            action = model.get_action(state)
        action_np = action.cpu().numpy()
        next_state, reward, done, truncated, _info = env.step(action_np)
        next_state = torch.tensor(next_state)
        model.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        if len(model.replay_buffer) > batch_size:
            train_res = model.train(batch_size=batch_size)
            if train_res is not None:
                actor_loss, critic_loss = train_res
                actor_loss_ep += actor_loss
                critic_loss_ep += critic_loss
                train_count += 1

    reward_history.append(ep_reward)
    avg_reward = np.mean(reward_history[-100:])
    if avg_reward > best_reward:
        best_reward = avg_reward
        model.save_checkpoint("checkpoints/bipedal_walker_td3")

    print("\nEpisode : ", episodes, " Episode Reward : %.1f"
          % ep_reward, " Avg Reward : %.1f" % avg_reward)
    if train_count > 0:
        logger.add_scalar("Episode Reward", ep_reward, episodes)
        logger.add_scalar("Avg Actor Loss", actor_loss_ep /
                          train_count, episodes)
        logger.add_scalar("Avg Critic Loss",
                          critic_loss_ep/train_count, episodes)
        logger.add_scalar("Avg Reward", avg_reward, episodes)
    if episodes % chekpt_interval == 0:
        model.save_checkpoint("checkpoints/bipedal_walker_td3")

env.close()

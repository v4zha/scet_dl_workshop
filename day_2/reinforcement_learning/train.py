import gym
import torch
import numpy as np
import pybullet_envs
from tqdm import tqdm
from scet_rl import TD3


resume_training = False
env = gym.make("Walker2DBulletEnv-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action_val = env.action_space.high[0]
agent = TD3(state_dim=state_dim, action_dim=action_dim, max_action_val=max_action_val,
            checkpt_file_path="checkpoints/walker2dbullet_td3", load_checkpoint=resume_training, actor_lr=1e-3, critic_lr=1e-3, log_dir="logs/walker2dbullet_td3_train")
print("device : ", agent.device)
logger = agent.writer
chekpt_interval = 100
batch_size = 100
max_episodes = 2000
warmup_steps = 3000 if not resume_training else 0
initial_steps = 0
reward_history = []

for ep in tqdm(range(max_episodes+1)):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    actor_loss_ep = 0
    critic_loss_ep = 0
    ep_reward = 0
    train_count = 0
    done = False
    while not done:
        if initial_steps < warmup_steps:
            action = env.action_space.sample()
            action = torch.tensor(action, dtype=torch.float32)
            initial_steps += 1
        else:
            action = agent.get_action(state)
        if initial_steps == warmup_steps:
            print("Warmup done")
            initial_steps += 1

        action_np = action.cpu().numpy()
        next_state, reward, done, _info = env.step(action_np)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward

        if len(agent.replay_buffer) > batch_size:
            train_res = agent.train(batch_size=batch_size)
            if train_res is not None:
                actor_loss, critic_loss = train_res
                actor_loss_ep += actor_loss
                critic_loss_ep += critic_loss
                train_count += 1

    reward_history.append(ep_reward)
    avg_reward = np.mean(reward_history[-100:])

    print("\nEpisode : ", ep, " Episode Reward : %.1f" %
          ep_reward, " Avg Reward : %.1f" % avg_reward)
    if train_count > 0:
        logger.add_scalar("Episode Reward", ep_reward, ep)
        logger.add_scalar("Avg Reward", avg_reward, ep)
        logger.add_scalar("actor_loss", actor_loss_ep/train_count, ep)
        logger.add_scalar("critic_loss", critic_loss_ep/train_count, ep)
    if ep % chekpt_interval == 0:
        agent.save_checkpoint("checkpoints/walker2dbullet_td3")

env.close()

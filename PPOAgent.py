import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium.wrappers import FrameStackObservation, ResizeObservation, GrayscaleObservation, NormalizeObservation
import matplotlib.pyplot as plt
import os
import datetime

from DQNAgent import DiscreteActionWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUSTOM_ACTIONS = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  # Steer, Full Gas, Light Brake
    (-1, 1,   0), (0, 1,   0), (1, 1,   0),  # Steer, Full Gas, No Brake  (Action 4 is (0,1,0) - straight, gas)
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),  # Steer, No Gas, Light Brake
    (-1, 0,   0), (0, 0,   0), (1, 0,   0)   # Steer, No Gas, No Brake
]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)

        # wyjscie aktora
        self.actor = nn.Linear(256, action_dim)

        #wyjscie krytyka
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = torch.relu(self.fc1(x))
        return self.actor(x), self.critic(x)


class PPOAgent:
    def __init__(self, env, image_size, frame_stack_size, clip_epsilon=0.2, gamma=0.99, lr=3e-4):
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episodes = 1000

        self.n_actions = env.action_space.n
        self.n_observation = frame_stack_size
        self.policy = ActorCritic(self.n_observation, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.memory = deque(maxlen=10000)
        self.name = "ppo_agent"  # Agent name for saving models
        self.gas_reward_bonus = 0.1  # Dodatkowa nagroda za gazowanie
        self.no_positive_reward_patience = 100  # Max consecutive steps without positive reward before truncating

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits, _ = self.policy(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action, probs.squeeze(0)[action].item()

    def store_transition(self, state, action, reward, next_state, done, prob):
        self.memory.append((state, action, reward, next_state, done, prob))

    def compute_returns(self, rewards, dones):
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)
            returns.insert(0, G)
        return returns

    def numpy_to_tensor(self, batch):
        states, actions, rewards, next_states, dones, old_probs = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        old_probs = torch.tensor(old_probs).to(self.device)
        return states, actions, rewards, next_states, dones, old_probs
    def learn(self):
        if len(self.memory) < 64:
            return

        batch = random.sample(self.memory, 64)
        states, actions, rewards, next_states, dones, old_probs = self.numpy_to_tensor(batch)

        #policzenie returnu
        returns = self.compute_returns(rewards, dones)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        #policzenie advantage
        _, values = self.policy(states)
        advantages = returns - values.squeeze()

        #aktualizacja polityki
        logits, values = self.policy(states)
        probs = torch.softmax(logits, dim=-1)
        new_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()

        ratio = new_probs / old_probs
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        value_loss = nn.MSELoss()(values.squeeze(), returns)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        rewards_per_episode = []
        gas_action_index = 4
        for episode in range(self.episodes):
            state, _ = self.env.reset(seed=1, options={"randomize": False})
            episode_reward = 0
            done = False
            consecutive_no_positive_reward = 0  # Counter for steps without positive reward
            truncated = False  # wlasna flaga
            while not done:
                action, prob = self.select_action(state)
                next_state, reward, terminated, truncated_env, _ = self.env.step(action)
                done = terminated or truncated

                if action == gas_action_index: # Assuming action 3 is 'gas'
                    reward += self.gas_reward_bonus

                # Check for positive reward
                if float(reward) > 0:
                    consecutive_no_positive_reward = 0
                else:
                    consecutive_no_positive_reward += 1

                if consecutive_no_positive_reward >= self.no_positive_reward_patience:
                    print(f"Episode {episode + 1} truncated early after {consecutive_no_positive_reward} steps without positive reward.")
                    truncated = True # Set our custom truncation flag
                    done = True # End the episode

                self.store_transition(state, action, reward, next_state, done, prob)
                state = next_state
                episode_reward += reward
            rewards_per_episode.append(episode_reward)
            self.learn()  # Dopiero po całym epizodzie
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        env.close()

        plt.plot(rewards_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Progress")
        plt.show()

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or test a PPO agent for Car Racing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--render', action='store_true', default=True,
                        help='Render the environment')
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='Load model from previous run for training or testing')
    args = parser.parse_args()

    # Print device info
    print(f"Using device: {device}")

    # Initialize with continuous=True to use our custom action mapping
    render_mode = "human" if args.render else "rgb_array"
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)

    # Record video every 5 episodes
    if render_mode != "human":
        env = gym.wrappers.RecordVideo(env, video_folder="videos/", episode_trigger=lambda ep: ep % 5 == 0)
    env = DiscreteActionWrapper(env, CUSTOM_ACTIONS)
    # Preprocessing
    image_size = 84
    frame_stack_size = 4
    env = ResizeObservation(env, (image_size, image_size))
    env = GrayscaleObservation(env)
    env = NormalizeObservation(env)
    env = FrameStackObservation(env, frame_stack_size)

    # Create agent
    agent = PPOAgent(env, image_size, frame_stack_size)

    # Setup saving path
    date = 'run-{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now()).replace(':', '-')
    save_path_dir = os.path.join('saved_models', 'car_racing', agent.name, date)
    save_path = os.path.join(save_path_dir, 'model.pt')

    # Load model if requested or in test mode
   #
    print(f"Env name: {env.__class__.__name__}")
    print(f"Mode: {args.mode}")

    # Train or test the agent
    agent.train()
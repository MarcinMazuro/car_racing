import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)

        # actor output
        self.actor = nn.Linear(256, action_dim)

        # critic output
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = torch.relu(self.fc1(x))
        return self.actor(x), self.critic(x)


class PPOAgent:
    def __init__(self, env, image_size, frame_stack_size, clip_epsilon=0.2, gamma=0.99, lr=3e-4, lam=0.95, epochs=10, minibatch_size=64):
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        self.lam = lam  # lambda for GAE
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episodes = 1000

        self.n_actions = env.action_space.n
        self.n_observation = frame_stack_size
        self.policy = ActorCritic(self.n_observation, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.memory = []
        self.name = "ppo_agent"
        self.gas_reward_bonus = 0.1
        self.no_positive_reward_patience = 300
        self.episode_counter = 0
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01

    def select_action(self, state, deterministic=False):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits, _ = self.policy(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
            prob = probs.squeeze(0)[action].item()
        else:
            action = torch.multinomial(probs, 1).item()
            prob = probs.squeeze(0)[action].item()
        return action, prob

    def store_transition(self, transition):
        self.memory.append(transition)

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def learn(self):
        # Prepare batch
        states, actions, rewards, dones, old_probs, values, next_states = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        old_probs = torch.tensor(old_probs, dtype=torch.float32).to(self.device)
        dones = np.array(dones, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        values = list(values)
        with torch.no_grad():
            # Ensure next_state is correctly shaped for the policy network
            # If memory can be empty or last transition doesn't have a valid next_state, handle appropriately
            if len(self.memory) > 0 and self.memory[-1][6] is not None:
                next_state_np = np.array(self.memory[-1][6])
                # Add batch dimension if it's a single state
                if next_state_np.ndim == len(states.shape) -1 : # Assuming states is (N, C, H, W) and next_state_np is (C,H,W)
                     next_state_np = np.expand_dims(next_state_np, axis=0)
                next_state_tensor = torch.tensor(next_state_np, dtype=torch.float32).to(self.device)
                _, next_value_tensor = self.policy(next_state_tensor)
                next_value = next_value_tensor.item()
            else: # Fallback if no next_state (e.g. memory empty or last state was terminal with no next_state stored)
                next_value = 0.0


        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # Normalize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)
        for _ in range(self.epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_probs = old_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logits, value = self.policy(mb_states)
                probs = torch.softmax(logits, dim=-1)
                new_probs_dist = torch.distributions.Categorical(probs) # For entropy calculation
                entropy = new_probs_dist.entropy().mean() # Calculate entropy

                new_probs_selected_action = probs.gather(1, mb_actions.unsqueeze(1)).squeeze()

                ratio = new_probs_selected_action / mb_old_probs.clamp(min=1e-8) # clamp old_probs to avoid division by zero
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                value_loss = nn.MSELoss()(value.squeeze(), mb_returns)
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.memory = []

    def rollout(self, max_steps=2048, deterministic=False):
        # Reset the environment at the start of the rollout
        state, _ = self.env.reset()
        done = False
        steps = 0
        episode_reward = 0
        consecutive_no_positive_reward = 0
        truncated = False
        rewards_per_episode = []
        gas_action_index = 4
        single_episode_rewards = []
        while steps < max_steps:
            value = self.policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))[1].item()
            action, prob = self.select_action(state, deterministic=deterministic)
            next_state, reward, terminated, truncated_env, _ = self.env.step(action)
            if action == gas_action_index:
                reward += self.gas_reward_bonus
            if float(reward) > 0:
                consecutive_no_positive_reward = 0
            else:
                consecutive_no_positive_reward += 1
            done = terminated or truncated_env
            if consecutive_no_positive_reward >= self.no_positive_reward_patience:
                truncated = True
                done = True
            self.store_transition((state, action, reward, done, prob, value, next_state))
            state = next_state
            episode_reward += reward
            steps += 1
            if done:
                rewards_per_episode.append(episode_reward)
                single_episode_rewards.append(episode_reward)
                self.episode_counter += 1 
                state, _ = self.env.reset()
                episode_reward = 0
                consecutive_no_positive_reward = 0
                truncated = False
        return single_episode_rewards

    def train(self, checkpoint_save_path):
        all_rewards = []
        rewards_file = os.path.join("rewards_ppo.txt")
        
        # Ensure the directory for the checkpoint_save_path exists
        os.makedirs(os.path.dirname(checkpoint_save_path), exist_ok=True)

        print(f"Starting training. Checkpoints will be saved to: {checkpoint_save_path}")

        for ppo_iter_idx in range(self.episodes): # self.episodes is the total number of PPO iterations
            rewards_in_rollout = self.rollout() # Collect experiences. self.episode_counter (if used for env episodes) increments inside rollout.
            all_rewards.extend(rewards_in_rollout)

            if not self.memory:
                print(f"PPO Iteration {ppo_iter_idx + 1}/{self.episodes}: No transitions in memory after rollout. Skipping learn step.")
                if not rewards_in_rollout: 
                    print(f"PPO Iteration {ppo_iter_idx + 1}/{self.episodes}: No full episodes completed in this rollout.")
                continue

            self.learn() # Update policy
            with open(rewards_file, "a") as f:
                for ep_reward in rewards_in_rollout:
                    f.write(f"{ep_reward}\n")

            current_ppo_iter_reward_sum = np.sum(rewards_in_rollout) if rewards_in_rollout else 0.0

            print(f"PPO Iteration {ppo_iter_idx + 1}/{self.episodes}: Reward Sum for this iteration = {current_ppo_iter_reward_sum:.2f}, Num Env Episodes in Rollout = {len(rewards_in_rollout)}")

            # Save model every 20 PPO iterations
            if (ppo_iter_idx + 1) % 20 == 0:
                self.save(checkpoint_save_path, extra_data={'ppo_iteration': ppo_iter_idx + 1, 'last_ppo_iter_reward_sum': current_ppo_iter_reward_sum})
        
        # Save one final time at the end of training
        self.save(checkpoint_save_path, extra_data={'ppo_iteration': self.episodes, 'last_ppo_iter_reward_sum': current_ppo_iter_reward_sum, 'status': 'training_completed'})
        print(f"Training finished. Final model saved to {checkpoint_save_path}")

        self.env.close()
        plt.plot(all_rewards) # This plots rewards from each full environment episode
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Progress")
        plt.show()

    def test(self, episodes=10):
        self.policy.eval()
        rewards_per_episode = []
        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated_env, _ = self.env.step(action)
                done = terminated or truncated_env
                state = next_state
                episode_reward += reward
            rewards_per_episode.append(episode_reward)
            print(f"Test Episode {ep+1}: Reward = {episode_reward:.2f}")
            with open("rewards_ppo.txt", "a") as f:
                f.write(f"{episode_reward}\n")

        self.env.close()
        plt.plot(rewards_per_episode)
        plt.xlabel("Test Episode")
        plt.ylabel("Reward")
        plt.title("Test Results")
        plt.show()

    def save(self, save_path, extra_data=None):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_dict = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gamma': self.gamma,
            'clip_epsilon': self.clip_epsilon,
            'lr': self.lr,
            'episodes': self.episodes,
            # 'memory': self.memory,  # Uncomment if you want to save memory
        }
        if extra_data:
            save_dict.update(extra_data)
        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No model found at {load_path}")
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.clip_epsilon = checkpoint.get('clip_epsilon', self.clip_epsilon)
        self.lr = checkpoint.get('lr', self.lr)
        self.episodes = checkpoint.get('episodes', self.episodes)
        print(f"Model loaded from {load_path}")


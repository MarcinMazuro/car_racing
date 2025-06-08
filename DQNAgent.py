import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, env, image_size, frame_stack_size):
        #hiperparametry
        self.learning_rate = 0.00025 #tempo uczenia modelu
        self.gamma = 0.99 #im wieksze tym bardziej dalekowzroczny model
        self.epsilon = 1.0 #poczatkowa wartosc epsilon (eksploracja)
        self.epsilon_min = 0.01#koniec eksploracji
        self.epsilon_decay = 0.01 #spadek wartosci epsilon
        self.batch_size = 64 #liczba doswiadczen do aktualizacji sieci
        self.target_update_freq = 1000 #co ile krokow aktualizowac siec docelowa
        self.memory_size = 10000 # rozmiar pamieci (liczba doswiadczen do przechowywania)
        self.episodes = 1000 #ilosc epizodow (prob jazdy)
        self.gas_reward_bonus = 0.1 # Dodatkowa nagroda za gazowanie
        self.no_positive_reward_patience = 100 # Max consecutive steps without positive reward before truncating
        self.name = "dqn_agent" # Agent name for saving models
        self.env = env

        self.n_actions = env.action_space.n #liczba mozliwych akcji do wykonaniwa (przyspiesz,hamuj, lewo, prawo, nic)
        self.n_observation = frame_stack_size #liczba obserwacji (4 klatki obrazu)
        self.policy_net = CNN(self.n_observation, self.n_actions).to(device) #siec uczaca sie ktora podejmuje decyzje
        self.target_net = CNN(self.n_observation, self.n_actions).to(device) #siec pomocnicza, ktora jest aktualizowana co pewien czas dla stabilnosci
        self.target_net.load_state_dict(self.policy_net.state_dict()) #kopiowanie wag z sieci uczacej do sieci docelowej
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=self.memory_size) #przychowywanie doswiadczen


    def select_action(self, state):
        if random.random() < self.epsilon:  #akcja losowa(eksploracja)
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state = state.to(device)
            q_values = self.policy_net(state)
            return q_values.argmax(1).item()   #akcja najlepsza mozliwa

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state), action, reward, np.array(next_state), done))

    def numpy_to_tensor(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)# element to transition(state, action, reward, next_state, done)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        return states, actions, rewards, next_states, dones

    def save(self, save_path, extra_data=None):
        """Save the model to the specified path"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'rewards_per_episode': getattr(self, 'rewards_per_episode', []),
            # 'memory': list(self.memory),  # Uncomment if you want to save replay buffer (can be large!)
        }
        if extra_data:
            save_dict.update(extra_data)
        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")

    def load(self, load_path):
        """Load the model from the specified path"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No model found at {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 1.0)
        self.epsilon_min = checkpoint.get('epsilon_min', 0.01)
        self.epsilon_decay = checkpoint.get('epsilon_decay', 0.01)
        self.rewards_per_episode = checkpoint.get('rewards_per_episode', [])
        # if 'memory' in checkpoint: self.memory = deque(checkpoint['memory'], maxlen=self.memory_size)
        print(f"Model loaded from {load_path}")

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size) #losujemy mini-batch z pamieci
        states, actions, rewards, next_states, dones = self.numpy_to_tensor(batch) #konwersja transition do tensora

        q_values = self.policy_net(states).gather(1, actions) #przepuszczenie stanu przez siec uczaca i pobranie wartosci Q dla akcji podanej w batchu

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)  #podajemy siec docelowa do przewidywania wartosci Q dla nastepnego stanu
            expected_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1)) #wzor bellmana

        loss = F.mse_loss(q_values, expected_q_values)  # 1.obliczenie błędu
        self.optimizer.zero_grad()  # 2. zerowanie gradientów
        loss.backward()  # 3. obliczanie gradientow
        self.optimizer.step()  #aktualizacja wag sieci


    def train(self, checkpoint_save_path):
        rewards_per_episode = []
        best_reward = float('-inf')
        steps_done = 0
        gas_action_index = 4 # Corresponds to CUSTOM_ACTIONS[4] == (0, 1, 0)

        # Setup saving path if not provided
        os.makedirs(os.path.dirname(checkpoint_save_path), exist_ok=True)
        print(f"Starting training. Checkpoints will be saved to: {checkpoint_save_path}")

        for episode in range(self.episodes):
            state,_ = self.env.reset(seed=1, options={"randomize": False}) # Set seed for consistent map
            episode_reward = 0
            skip_learn = 4 #co ile krokow uczymy model
            done = False
            truncated = False # Initialize truncated flag
            consecutive_no_positive_reward = 0 # Counter for steps without positive reward

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) #konwersja stanu do tensora
                action = self.select_action(state_tensor)#wybor akcji

                next_state, reward, terminated, truncated_env, _ = self.env.step(action)#wykonanie akcji
                # truncated_env is the truncation signal from the environment (e.g. time limit)
                # We use our own `truncated` for early stopping logic.

                # Add bonus reward for using gas
                if action == gas_action_index: # Assuming action 3 is 'gas'
                    reward += self.gas_reward_bonus

                # Check for positive reward
                if float(reward) > 0:
                    consecutive_no_positive_reward = 0
                else:
                    consecutive_no_positive_reward += 1

                # Original done condition from environment
                done = terminated or truncated_env

                # Early stopping condition
                if consecutive_no_positive_reward >= self.no_positive_reward_patience:
                    print(f"Episode {episode + 1} truncated early after {consecutive_no_positive_reward} steps without positive reward.")
                    truncated = True # Set our custom truncation flag
                    done = True # End the episode

                self.store_transition(state, action, reward, next_state, done) #zapisanie doswiadczenia do pamieci

                state = next_state
                episode_reward += reward
                if steps_done % skip_learn == 0: # co x krokow uczymy model, dla szybkosci
                    self.learn()

                if steps_done % self.target_update_freq == 0: # aktualizujemy siec docelowa
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                steps_done += 1

            if self.epsilon > self.epsilon_min: # zmniejszamy wartosc epsilon
                self.epsilon -= self.epsilon_decay

            rewards_per_episode.append(episode_reward)
            self.rewards_per_episode = rewards_per_episode  # Save for checkpointing

            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {self.epsilon:.3f}")

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                if checkpoint_save_path:
                    self.save(checkpoint_save_path, extra_data={'best_reward': best_reward, 'episode': episode + 1})

            # (Opcjonalnie) Save after each episode
            # if save_path:
            #     self.save(save_path)

        self.env.close()

        plt.plot(rewards_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Progress")
        plt.show()

    def test(self, episodes=10):
        self.policy_net.eval()
        rewards_per_episode = []
        self.epsilon = 0.0
        self.epsilon_min = 0.0
        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(
                    device)
                action = self.select_action(state_tensor)
                next_state, reward, terminated, truncated_env, _ = self.env.step(action)
                done = terminated or truncated_env
                state = next_state
                episode_reward += reward
            rewards_per_episode.append(episode_reward)
            print(f"Test Episode {ep+1}: Reward = {episode_reward:.2f}")
        self.env.close()
        plt.plot(rewards_per_episode)
        plt.xlabel("Test Episode")
        plt.ylabel("Reward")
        plt.title("Test Results")
        plt.show()




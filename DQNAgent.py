import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from gymnasium.wrappers import FrameStackObservation, ResizeObservation, GrayscaleObservation, NormalizeObservation
import matplotlib.pyplot as plt


class CNN(nn.Module):    #tej sieci troche nie czaje
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
        self.episodes = 10 #ilosc epizodow (prob jazdy)

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

    def numpy_to_tensor(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        return states, actions, rewards, next_states, dones

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size) #losujemy mini-batch z pamieci
        states, actions, rewards, next_states, dones = zip(*batch)# element to transition(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = self.numpy_to_tensor(states, actions, rewards, next_states, dones) #konwersja transition do tensora

        q_values = self.policy_net(states).gather(1, actions) #przepuszczenie stanu przez siec uczaca i pobranie wartosci Q dla akcji podanej w batchu

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)  #podajemy siec docelowa do przewidywania wartosci Q dla nastepnego stanu
            expected_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1)) #wzor bellmana

        loss = F.mse_loss(q_values, expected_q_values)  # 1.obliczenie błędu
        self.optimizer.zero_grad()  # 2. zerowanie gradientów
        loss.backward()  # 3. obliczanie gradientow
        self.optimizer.step()  #aktualizacja wag sieci


    def train(self):
        rewards_per_episode = []
        steps_done = 0

        for episode in range(self.episodes):
            state,_ = env.reset(options={"randomize": False})
            episode_reward = 0
            skip_learn = 4 #co ile krokow uczymy model
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) #konwersja stanu do tensora
                action = self.select_action(state_tensor)#wybor akcji

                next_state, reward, terminated, truncated, _ = env.step(action)#wykonanie akcji
                done = terminated or truncated

                self.store_transition(state, action, reward, next_state, done) #zapisanie doswiadczenia do pamieci

                state = next_state
                episode_reward += reward
                if steps_done % skip_learn == 0: # co 10 krokow uczymy model, dla szybkosci
                    self.learn()

                if steps_done % self.target_update_freq == 0: # aktualizujemy siec docelowa
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                steps_done += 1

            if self.epsilon > self.epsilon_min: # zmniejszamy wartosc epsilon
                self.epsilon -= self.epsilon_decay

            rewards_per_episode.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {self.epsilon:.3f}")

        env.close()

        plt.plot(rewards_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Progress")
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    env = gym.wrappers.RecordVideo(env, video_folder="videos/", episode_trigger=lambda ep: ep % 5 == 0) #co ile epizodow nagrac film

    #preprocessing
    image_size = 84
    frame_stack_size = 4 # do stworzenia obserwacji jako 4 ostatnie klatki, aby moc porownywac ruch pojazdu
    env = ResizeObservation(env, (image_size, image_size)) #zmiana rozmiaru obserwacji do 96x96
    env = GrayscaleObservation(env) # zamiana obserwacji na skale szarości
    env = NormalizeObservation(env)  # normalizacja obserwacji
    env = FrameStackObservation(env, frame_stack_size)  #stworzenie obserwacji jako 4 ostatnie klatki

    agent = DQNAgent(env, image_size, frame_stack_size)
    agent.train()

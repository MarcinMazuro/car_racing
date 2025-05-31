import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import cv2


def image_preprocessing(img):
    #przetwarzanie klatki
    img = cv2.resize(img, dsize=(84, 84)) #zmiana rozmiaru z 96x96 na 84x84
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0 # normalizacja i konwersja do skali szarości
    return img


# DQN Network
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
    def __init__(self):
        #hiperparametry
        self.learning_rate = 0.005 #tempo uczenia modelu
        self.gamma = 0.99 #im wieksze tym bardziej dalekowzroczny model
        self.epsilon = 1.0 #poczatkowa wartosc epsilon (eksploracja)
        self.epsilon_min = 0.05#koniec eksploracji
        self.epsilon_decay = 0.01 #spadek wartosci epsilon
        self.batch_size = 64 #liczba doswiadczen do aktualizacji sieci
        self.target_update_freq = 1000 #co ile krokow aktualizowac siec docelowa
        self.memory_size = 10000 # rozmiar pamieci (liczba doswiadczen do przechowywania)
        self.episodes = 10 #ilosc epizodow (prob jazdy)

        self.n_actions = 5 #liczba mozliwych akcji do wykonaniwa (przyspiesz,hamuj, lewo, prawo, nic)
        self.n_observation = 4 #liczba obserwacji (4 klatki obrazu)
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

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size) #losujemy mini-batch z pamieci
                                                            # element to transition(state, action, reward, next_state, done)

        state_batch = torch.cat([transition[0] for transition in batch]).to(device, dtype=torch.float32)
        action_batch = torch.tensor([transition[1] for transition in batch], device=device).unsqueeze(1)
        reward_batch = torch.tensor([transition[2] for transition in batch], device=device, dtype=torch.float32)
        next_state_batch = torch.cat([transition[3] for transition in batch]).to(device, dtype=torch.float32)
        done_batch = torch.tensor([transition[4] for transition in batch], device=device, dtype=torch.float32)

        q_values = self.policy_net(state_batch).gather(1, action_batch) #przepuszczenie stanu przez siec uczaca i pobranie wartosci Q dla akcji podanej w batchu

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0] #podajemy siec docelowa do przewidywania wartosci Q dla nastepnego stanu
            expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch) #wzor bellmana

        loss = F.mse_loss(q_values.squeeze(), expected_q_values)  # 1.obliczenie błędu
        self.optimizer.zero_grad()  # 2. zerowanie gradientów
        loss.backward()  # 3. obliczanie gradientow
        self.optimizer.step()  #aktualizacja wag sieci

    def train(self):
        rewards_per_episode = []
        steps_done = 0
        current_epsilon = self.epsilon

        for episode in range(self.episodes):
            state,_ = env.reset(options={"randomize": False})
            frame = image_preprocessing(state) #przetwarzanie pierwszej klatki
            state = np.stack([frame] * 4, axis=0) #stworzenie stanu jako 4 ostatnie klatki (4x84x84)
            episode_reward = 0
            skip_steps = 10
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) #konwersja stanu do tensora
                action_idx = self.select_action(state_tensor)#wybor akcji

                next_obs, reward, terminated, truncated, _ = env.step(action_idx)#wykonanie akcji
                done = terminated or truncated

                next_frame = image_preprocessing(next_obs)  # [84, 84]
                #stan (observation space) to ostatnie 4 klatki o rozmiarze 84x84
                #nalezy wiec dodac nowa klatke do stanu i usunać najstarsza
                next_state = np.concatenate((state[1:], next_frame[np.newaxis]), axis=0)

                self.memory.append((torch.tensor(state).unsqueeze(0), #zapisanie do pamieci doswiadczenia
                               action_idx,
                               reward,
                               torch.tensor(next_state).unsqueeze(0),
                               float(done)))

                state = next_state
                episode_reward += reward

                if steps_done % skip_steps == 0: # co 10 krokow uczymy model, dla szybkosci
                    self.learn()
                    env.render()

                steps_done += 1
                if steps_done % self.target_update_freq == 0: # aktualizujemy siec docelowa
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            if current_epsilon > self.epsilon_min: # zmniejszamy wartosc epsilon
                current_epsilon -= self.epsilon_decay

            rewards_per_episode.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {current_epsilon:.3f}")

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

    agent = DQNAgent()
    agent.train()


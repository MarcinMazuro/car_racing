import matplotlib.pyplot as plt

# Funkcja do wczytywania nagród z pliku
def load_rewards(path):
    with open(path) as f:
        return [float(line.strip()) for line in f if line.strip()]

# Funkcja do obliczania średniej kroczącej
def moving_average(data, window_size=20):
    return [sum(data[max(0, i - window_size + 1):i + 1]) / (i - max(0, i - window_size + 1) + 1) for i in range(len(data))]

# Wczytaj dane
ppo_rewards = load_rewards('rewards_ppo.txt')
dqn_rewards = load_rewards('rewards_dqn.txt')

# Oblicz średnie i minima
ppo_mean = sum(ppo_rewards) / len(ppo_rewards)
ppo_max = max(ppo_rewards)
dqn_mean = sum(dqn_rewards) / len(dqn_rewards)
dqn_max = max(dqn_rewards)

# Oblicz średnie kroczące
ppo_ma = moving_average(ppo_rewards, window_size=20)
dqn_ma = moving_average(dqn_rewards, window_size=20)

# Wyświetl statystyki
print(f"PPO - Średnia: {ppo_mean:.2f}, Max: {ppo_max:.2f}")
print(f"DQN - Średnia: {dqn_mean:.2f}, Max: {dqn_max:.2f}")

# Wykres
plt.figure(figsize=(12, 6))
plt.plot(ppo_ma, label=f'PPO (moving average 20) Średnia: {ppo_mean:.2f}, Max: {ppo_max:.2f}', color='blue')
plt.plot(dqn_ma, label=f'DQN (moving average 20) Średnia: {dqn_mean:.2f}, Max: {dqn_max:.2f}', color='green')
plt.title("Train: PPO vs DQN")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

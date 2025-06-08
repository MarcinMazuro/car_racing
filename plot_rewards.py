import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read data from the file
file_path = "c:\\Nauka\\projekty\\car_racing\\rewards.txt"
with open(file_path, 'r') as file:
    rewards = [float(line.strip()) for line in file if line.strip() and not line.strip().startswith('//')]

# Convert to numpy array and create index array
rewards = np.array(rewards)
# Divide all rewards by 2
rewards = rewards / 2
indices = np.arange(len(rewards))

# Calculate moving average (window size can be adjusted)
window_size = 20
moving_avg = pd.Series(rewards).rolling(window=window_size).mean().to_numpy()

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(indices, rewards, '-', alpha=0.5)
plt.plot(indices, moving_avg, 'r-', linewidth=2, label=f'Moving average (window={window_size})')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training progress')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("c:\\Nauka\\projekty\\car_racing\\rewards_plot.png")
plt.show()

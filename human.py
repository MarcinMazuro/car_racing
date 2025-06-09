import gymnasium as gym
import pygame
import numpy as np

# Initialize the environment
env = gym.make("CarRacing-v3", render_mode="human", domain_randomize = False)
observation, info = env.reset()

# Initialize pygame for keyboard handling
pygame.init()
screen = pygame.display
pygame.display.set_caption("CarRacing Control")

# Key mapping to actions
action = np.array([0.0, 0.0, 0.0])  # [steer, gas, brake]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Reset actions
    steer = 0.0
    gas = 0.0
    brake = 0.0

    if keys[pygame.K_LEFT]:
        steer = -1.0
    elif keys[pygame.K_RIGHT]:
        steer = 1.0

    if keys[pygame.K_UP]:
        gas = 1.0
    elif keys[pygame.K_DOWN]:
        brake = 0.8

    action = np.array([steer, gas, brake])

    # Perform action in the environment
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("End of episode. Restarting...")
        observation, info = env.reset(options={"randomize": False})

    # Small delay for loop stability
    pygame.time.wait(30)

env.close()
pygame.quit()

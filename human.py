import gymnasium as gym
import pygame
import numpy as np

# Inicjalizacja środowiska
env = gym.make("CarRacing-v3", render_mode="human", domain_randomize = False)
observation, info = env.reset()

# Inicjalizacja pygame do obsługi klawiatury
pygame.init()
screen = pygame.display
pygame.display.set_caption("Sterowanie CarRacing")

# Mapowanie klawiszy na akcje
action = np.array([0.0, 0.0, 0.0])  # [steer, gas, brake]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Reset akcji
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

    # Wykonaj akcję w środowisku
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Koniec epizodu. Restart...")
        observation, info = env.reset(options={"randomize": False})

    # Mała przerwa dla stabilności pętli
    pygame.time.wait(30)

env.close()
pygame.quit()

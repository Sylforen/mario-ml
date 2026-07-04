import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import pygame
import numpy as np

env = gym_super_mario_bros.make(
    "SuperMarioBros-v0",
    render_mode="rgb_array"
)

env = JoypadSpace(env, SIMPLE_MOVEMENT)

pygame.init()

obs, info = env.reset()

screen = None
clock = pygame.time.Clock()

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    frame = env.render()  # (240, 256, 3) RGB

    if screen is None:
        h, w, _ = frame.shape
        screen = pygame.display.set_mode((w, h))

    # pygame expects (width, height, 3)
    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))

    screen.blit(surf, (0, 0))

    # Example overlay (you can draw ML features here)
    pygame.draw.circle(screen, (255, 0, 0), (50, 50), 5)

    pygame.display.flip()
    clock.tick(60)

env.close()
pygame.quit()
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import pygame
import numpy as np
from perception import SpriteRegistry, PerceptionEngine

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


# perception

registry = SpriteRegistry("feature-matching/sprite_registry.yaml")
perception = PerceptionEngine(registry)

#

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    frame = env.render()

    # init screen FIRST
    if screen is None:
        h, w, _ = frame.shape
        screen = pygame.display.set_mode((w, h))

    # convert frame
    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))

    # draw base frame first
    screen.blit(surf, (0, 0))

    # perception
    entities = perception.detect(frame)

    # overlays AFTER background
    for e in entities:
        x, y, w, h = e.bbox
        pygame.draw.rect(screen, (0, 255, 0), (x, y, w, h), 5)

    # debug marker
    pygame.draw.circle(screen, (255, 0, 0), (50, 50), 20)

    pygame.display.flip()
    clock.tick(60)
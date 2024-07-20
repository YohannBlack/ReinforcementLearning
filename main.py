import pygame
import numpy as np

from environments import LineWorld, GridWorld
from algorithms import value_iteration, policy_iteration, naive_monte_carlo_with_exploring_starts


def line_world():
    env = LineWorld(length=10)
    # policy, _ = value_iteration(env)
    # policy, _ = policy_iteration(env)
    policy = naive_monte_carlo_with_exploring_starts(env, nb_iter=10000)
    print(policy)

    pygame.init()
    screen = pygame.display.set_mode((env.length * 50, 50))
    pygame.display.set_caption('LineWorld')

    running = True
    state = env.reset()
    clock = pygame.time.Clock()

    print(policy[state])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not isinstance(policy, dict):
            action = np.argmax(policy[state])
            state, _, done = env.step(env.actions[action])
            env.render(screen)
            clock.tick(5)
        else:
            action = policy[state]
            state, _, done = env.step(action)
            env.render(screen)
            clock.tick(5)

        if done:
            state = env.reset()

    pygame.quit()

def grid_world():
    env = GridWorld(width=5, height=5)
    # policy, _ = value_iteration(env)
    # policy, _ = policy_iteration(env)
    policy = naive_monte_carlo_with_exploring_starts(env, nb_iter=10000)

    print(policy)

    pygame.init()
    screen_size = (env.width * 100, env.height * 100)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption('GridWorld')

    running = True
    state = env.reset()
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not isinstance(policy, dict):
            action = np.argmax(policy[state])
            state, _, done = env.step(env.actions[action])
            env.render(screen)
            clock.tick(5)
        else:
            action = policy[state]
            print(state, action)
            state, _, done = env.step(action)
            env.render(screen)
            clock.tick(5)

        if done:
            state = env.reset()
    
    pygame.quit()


if __name__ == "__main__":
    line_world()
    grid_world()
import pygame
import numpy as np
import pprint

from environments import LineWorld, GridWorld
from algorithms import value_iteration, policy_iteration, monte_carlo_with_exploring_start, monte_carlo_on_policy, monte_carlo_off_policy


def line_world():
    env = LineWorld(length=10)
    # policy, V, episode = value_iteration(env)
    # policy, V, episode = policy_iteration(env)
    # policy, Q = monte_carlo_with_exploring_start(env, nb_iter=1000, GAMMA=0.7)
    # policy, Q = monte_carlo_on_policy(env, nb_iter=10000)
    policy, Q = monte_carlo_off_policy(env, nb_iter=10000)

    if isinstance(policy, dict):
        print("Optimal policy: \n")
        pprint.pprint(policy)
    else:
        print("Optimal policy: \n", policy)

    pygame.init()
    screen = pygame.display.set_mode((env.length * 50, 50))
    pygame.display.set_caption('LineWorld')

    running = True
    state = env.reset()
    clock = pygame.time.Clock()


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = np.argmax(policy[state])
        state, _, done = env.step(env.actions[action])
        env.render(screen)
        clock.tick(5)

        if done:
            state = env.reset()

    pygame.quit()

def grid_world():
    env = GridWorld(width=5, height=5)
    # policy, V, episode = value_iteration(env, GAMMA=0.3)
    # policy, V, episode = policy_iteration(env, nb_iter=50000, GAMMA=0.3)
    # policy, Q = monte_carlo_with_exploring_start(
    #     env, nb_iter=10000, max_step=100, GAMMA=0.3)
    # policy, Q = monte_carlo_on_policy(env, nb_iter=10000)
    policy, Q = monte_carlo_off_policy(env, nb_iter=10000)

    if isinstance(policy, dict):
        print("Optimal policy: \n")
        pprint.pprint(policy)
    else:
        print("Optimal policy: \n", policy)

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

        action = np.argmax(policy[state])
        state, _, done = env.step(env.actions[action])
        env.render(screen)
        clock.tick(5)

        if done:
            state = env.reset()
    
    pygame.quit()


if __name__ == "__main__":
    line_world()
    grid_world()

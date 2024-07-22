import pygame
import numpy as np
import pprint
import time

from secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3
from environments import LineWorld, GridWorld
from algorithms import value_iteration, policy_iteration, monte_carlo_with_exploring_start, monte_carlo_on_policy, monte_carlo_off_policy


def line_world():
    env = LineWorld(length=5)

    start = time.time()
    # policy, V, episode = value_iteration(env)
    policy, V, episode = policy_iteration(env)
    # policy, Q = monte_carlo_with_exploring_start(env, nb_iter=10000)
    # policy, Q = monte_carlo_on_policy(env, nb_iter=10000)
    # policy, Q = monte_carlo_off_policy(env, nb_iter=10000)
    print("Time taken to train: ", time.time() - start)

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
        env.step(env.actions[action])
        env.render(screen)
        clock.tick(5)

        if env.is_game_over():
            state = env.reset()

    pygame.quit()

def grid_world():
    env = GridWorld(width=5, height=5)

    start = time.time()
    # policy, V, episode = value_iteration(env, GAMMA=0.4)
    # policy, V, episode = policy_iteration(env, GAMMA=0.3)
    policy, Q = monte_carlo_with_exploring_start(
        env, nb_iter=10000, max_step=100, GAMMA=0.999)
    # policy, Q = monte_carlo_on_policy(env, nb_iter=10000)
    # policy, Q = monte_carlo_off_policy(env, nb_iter=10000)
    print("Time taken to train: ", time.time() - start)

    if isinstance(policy, dict):
        print("Optimal policy: \n")
        pprint.pprint(policy)
    else:
        print("Optimal policy: \n", policy)

    for state, action_prob in policy.items():
        print(state, np.argmax(action_prob))

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
        print(env.state_id(), action)
        env.step(action)
        env.render(screen)
        clock.tick(5)

        if env.is_game_over():
            return
    
    pygame.quit()


def secret_env0():
    env = SecretEnv0()

    start = time.time()
    # policy, V, episode = value_iteration(env)
    # policy, V, episode = policy_iteration(env)
    policy, Q = monte_carlo_with_exploring_start(
        env, nb_iter=10000, max_step=8000, GAMMA=0.9)
    # policy, Q = monte_carlo_on_policy(env, nb_iter=10000)
    # policy, Q = monte_carlo_off_policy(env, nb_iter=10000)
    print("Time taken to train: ", time.time() - start)

    if isinstance(policy, dict):
        print("Optimal policy: \n")
        pprint.pprint(policy)
    else:
        print("Optimal policy: \n", policy)
        pass

    reward = 0
    running = True
    while running:
        action = np.argmax(policy[env.state_id()])
        env.step(action)
        reward += env.score()

        if env.is_game_over():
            env.reset()
            print("Total reward: ", reward)
            running = False



if __name__ == "__main__":
    # line_world()
    grid_world()
    # secret_env0()

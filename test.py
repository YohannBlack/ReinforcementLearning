import pygame
import numpy as np
import pprint

import time
import matplotlib.pyplot as plt

from utils import *
from secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3
from environments import LineWorld, GridWorld
from algorithms import value_iteration, policy_iteration, monte_carlo_with_exploring_start, monte_carlo_on_policy, monte_carlo_off_policy, sarsa, expected_sarsa, q_learning, dyna_q


def line_world_vi(save=False, load=False, run=False, filename="line_world_vi_policy.json"):
    env = LineWorld(length=5)

    if not load:
        V, policy = value_iteration(env)
    else:
       V, policy = load_policy_dynamic(filename)

    if save:
        save_policy_dynamic(policy, V, filename)

    if run:
        pygame.init()
        screen = pygame.display.set_mode((env.length * 50, 50))
        pygame.display.set_caption('LineWorld')

        running = run
        state = env.reset()
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = policy[state]
            env.step(action)
            env.render(screen)
            clock.tick(5)

            if env.is_game_over():
                state = env.reset()

        pygame.quit()


def line_world_pi(save=False, load=False, run=False, filename="line_world_pi_policy.json"):
    env = LineWorld(length=5)

    if not load:
        V, policy = value_iteration(env)
    else:
       V, policy = load_policy_dynamic(filename)

    if save:
        save_policy_dynamic(policy, V, filename)

    if run:
        pygame.init()
        screen = pygame.display.set_mode((env.length * 50, 50))
        pygame.display.set_caption('LineWorld')

        running = run
        state = env.reset()
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = policy[state]
            env.step(env.actions[action])
            env.render(screen)
            clock.tick(5)

            if env.is_game_over():
                state = env.reset()

        pygame.quit()


def line_world_mc_es(save=False, load=False, run=False, filename="line_world_mc_es_policy.json", nb_iter=10000, max_step=100, GAMMA=0.999):
    env = LineWorld(length=5)

    if not load:
        policy, Q, cummul_avg_es, mean_Q_es = monte_carlo_with_exploring_start(
            env, nb_iter=nb_iter, max_step=max_step, GAMMA=GAMMA)
    else:
        policy, Q = load_mc_es(filename)

    if save:
        save_mc_es(policy, Q, filename)

    if run:
        pygame.init()
        screen = pygame.display.set_mode((env.length * 50, 50))
        pygame.display.set_caption('LineWorld')

        running = run
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


def line_world_mc_onp():
    env = LineWorld(length=5)

    start = time.time()
    policy, Q, cummul_avg_onp, mean_Q_onp = monte_carlo_on_policy(
        env, nb_iter=10000, max_step=100, GAMMA=0.999)
    print("Time taken to train: ", time.time() - start)

    print(policy)

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
        action = policy[state]
        env.step(action)

        env.render(screen)
        clock.tick(5)

        if env.is_game_over():
            state = env.reset()


def line_world_mc_offp():
    env = LineWorld(length=5)

    start = time.time()
    policy, Q, cummul_avg_ofp, mean_Q_ofp = monte_carlo_off_policy(
        env, nb_iter=10000, max_step=100, GAMMA=0.999)
    print("Time taken to train: ", time.time() - start)

    print(policy)

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
        action = max(policy[state], key=policy[state].get)
        env.step(env.actions[action])

        env.render(screen)
        clock.tick(5)

        if env.is_game_over():
            state = env.reset()

    pygame.quit()


def line_world_sarsa():
    env = LineWorld(length=5)

    start = time.time()
    Q, cummul_avg_sarsa = sarsa(env, nb_iter=10000, max_step=100, gamma=0.999)
    print("Time taken to train: ", time.time() - start)

    print(Q)
    policy = {}
    for state, actions in Q.items():
        best_action = np.argmax(actions)

        ont_hot_action = np.zeros(env.num_actions())
        ont_hot_action[best_action] = 1.0

        policy[state] = ont_hot_action

    print(policy)

    pygame.init()
    screen = pygame.display.set_mode((env.length * 50, 50))
    pygame.display.set_caption('LineWorld')

    env.reset()
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        action = np.argmax(policy[env.state_id()])
        env.step(action)
        env.render(screen)
        clock.tick(5)

        if env.is_game_over():
            state = env.reset()


def secret_env_0_sarsa():
    env = SecretEnv0()

    start = time.time()
    Q, cummul_avg_sarsa = sarsa(env, nb_iter=10000, max_step=100, gamma=0.999)
    print("Time taken to train: ", time.time() - start)

    print(Q)

    policy = {}
    for state, actions in Q.items():
        best_action = np.argmax(actions)

        ont_hot_action = np.zeros(env.num_actions())
        ont_hot_action[best_action] = 1.0

        policy[state] = ont_hot_action

    print(policy)

    reward = 0
    i = 0
    env.reset()
    running = True

    while running:
        action = np.argmax(policy[env.state_id()])
        env.step(action)
        reward += env.score()
        i += 1
        env.display()

        if env.is_game_over():
            running = False
            print("Total reward for secret env 0: ", reward)
            print("Number of steps: ", i)
            print("Number of state visited: ", len(policy.keys()))

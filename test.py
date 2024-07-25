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
        V, policy = policy_iteration(env)
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


def visualize_mc_line_world(nb_iter=10000, max_step=100, gamma=0.999):
    env = LineWorld(length=5)

    es, onp, offp, exec_time = execute_comparison_mc(
        env, nb_iter=nb_iter, max_step=max_step, gamma=gamma)
    visualize_monte_carlo(es, onp, offp, exec_time, "line_world")

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
            env.step(action)

            env.render(screen)
            clock.tick(5)

            if env.is_game_over():
                state = env.reset()

        pygame.quit()

def line_world_mc_onp(save=False, load=False, run=False, filename="line_world_mc_onp_policy.json", nb_iter=10000, max_step=100, GAMMA=0.999):
    env = LineWorld(length=5)

    if not load:
        policy, Q, cummul_avg_onp = monte_carlo_on_policy(
            env, nb_iter=10000, max_step=100, GAMMA=0.999)
    else:
        policy, Q = load_mc_onp(filename)

    if save:
        save_mc_onp(policy, Q, filename)

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


def line_world_mc_offp(save=False, load=False, run=False, filename="line_world_mc_offp_policy.json", nb_iter=10000, max_step=100, GAMMA=0.999):
    env = LineWorld(length=5)

    if not load:
        policy, Q, cummul_avg_ofp, mean_Q_ofp = monte_carlo_off_policy(
            env, nb_iter=nb_iter, max_step=max_step, gamma=GAMMA)
    else:
        policy, Q = load_mc_onp(filename)
        print(policy)
        print(Q)

    if save:
        save_mc_offp(policy, Q, filename)

    if run:
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
            action = max(policy[env.state_id()],
                         key=policy[env.state_id()].get)
            env.step(action)

            env.render(screen)
            clock.tick(5)

            if env.is_game_over():
                state = env.reset()

        pygame.quit()


def line_world_sarsa(save=False, load=False, run=False, filename="line_world_sarsa_policy.json", nb_iter=10000, max_step=100, gamma=0.999):
    env = LineWorld(length=5)

    if not load:
        Q, cummul_avg_sarsa = sarsa(
            env, nb_iter=nb_iter, max_step=max_step, gamma=gamma)
        policy = {}
        for state, actions in Q.items():
            best_action = np.argmax(actions)

            ont_hot_action = np.zeros(env.num_actions())
            ont_hot_action[best_action] = 1.0

            policy[state] = ont_hot_action
    else:
        policy, Q = load_sarsa(filename)

    if save:
        save_sarsa(policy, Q, filename)

    if run:
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


def line_world_q_learning(save=False, load=False, run=False, filename="line_world_q_learning_policy.json", nb_iter=10000, max_step=100, gamma=0.999, epsilon=0.1, alpha=0.5):
    env = LineWorld(length=5)

    if not load:
        Q, cummul_avg_q_l = q_learning(
            env, nb_iter=nb_iter, max_step=max_step, gamma=gamma, epsilon=epsilon, alpha=alpha)
        policy = {}
        for state, actions in Q.items():
            best_action = np.argmax(actions)

            ont_hot_action = np.zeros(env.num_actions())
            ont_hot_action[best_action] = 1.0

            policy[state] = ont_hot_action
    else:
        policy, Q = load_sarsa(filename)

    if save:
        save_sarsa(policy, Q, filename)

    if run:
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


def visualize_td_line_world(nb_iter=10000, max_step=100, gamma=0.999, alpha=0.1, epsilon=0.1):
    env = LineWorld(length=5)

    sarsa, q_l, exec_time = execute_comparison_td(
        env, nb_iter=nb_iter, alpha=alpha, gamma=gamma, epsilon=epsilon, max_step=max_step)
    visualize_temporal_difference(sarsa, q_l, exec_time, "line_world")

def line_world_dyna_q(save=False, load=False, run=False, filename="line_world_dyna_q_policy.json", nb_iter=10000, max_step=100, gamma=0.999, epsilon=0.1, alpha=0.1, n=10):
    env = LineWorld(length=5)

    if not load:
        Q, cummul_avg_q_l = dyna_q(
            env, nb_iter=nb_iter, max_step=max_step, gamma=gamma, epsilon=epsilon, alpha=alpha, planning_steps=n)
        policy = {}
        for state in range(len(Q)):
            best_action = np.argmax(Q[state])

            ont_hot_action = np.zeros(env.num_actions())
            ont_hot_action[best_action] = 1.0

            policy[state] = ont_hot_action
        print(policy)
    else:
        policy, Q = load_dyna_q(filename)

    if save:
        save_dyna_q(policy, Q, filename)

    if run:
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


def visualize_mc_secret_env_0(nb_iter=10000, max_step=100, gamma=0.999):
    env = SecretEnv0()

    es, onp, offp, exec_time = execute_comparison_mc(
        env, nb_iter=nb_iter, max_step=max_step, gamma=gamma)
    visualize_monte_carlo(es, onp, offp, exec_time, "env0")

def secret_env_0_mc_es(save=False, load=False, run=False, filename="secret_env0_mc_es_policy.json", nb_iter=10000, max_step=100, GAMMA=0.999):
    env = SecretEnv0()

    if not load:
        policy, Q, cummul_avg_es, mean_Q_es = monte_carlo_with_exploring_start(
            env, nb_iter=nb_iter, max_step=max_step, GAMMA=GAMMA)
    else:
        policy, Q = load_mc_es(filename)

    if save:
        save_mc_es(policy, Q, filename)

    env.reset()

    if run:
        reward = 0
        i = 0
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


def secret_env_0_mc_onp(save=False, load=False, run=False, filename="secret_env0_mc_onp_policy.json", nb_iter=10000, max_step=100, GAMMA=0.999):
    env = SecretEnv0()

    if not load:
        policy, Q, cummul_avg_onp = monte_carlo_on_policy(
            env, nb_iter=nb_iter, max_step=max_step, GAMMA=GAMMA)
    else:
        policy, Q = load_mc_onp(filename)

    if save:
        save_mc_onp(policy, Q, filename)

    if run:
        reward = 0
        i = 0
        running = run
        while running:
            action = policy[env.state_id()]
            env.step(action)
            reward += env.score()
            i += 1
            env.display()

            if env.is_game_over():
                running = False
                print("Total reward for secret env 0: ", reward)
                print("Number of steps: ", i)
                print("Number of state visited: ", len(policy.keys()))


def secret_env_0_mc_offp(save=False, load=False, run=False, filename="secret_env0_mc_offp_policy.json", nb_iter=10000, max_step=100, GAMMA=0.999):
    env = SecretEnv0()

    if not load:
        policy, Q, cummul_avg_ofp, mean_Q_ofp = monte_carlo_off_policy(
            env, nb_iter=nb_iter, max_step=max_step, gamma=GAMMA)
    else:
        policy, Q = load_mc_onp(filename)

    if save:
        save_mc_offp(policy, Q, filename)

    if run:
        env.reset()
        reward = 0
        i = 0
        running = run
        while running:
            action = np.argmax(policy[env.state_id()])
            env.step(action)

            if env.is_game_over():
                running = False
                print("Total reward for secret env 0: ", reward)
                print("Number of steps: ", i)
                print("Number of state visited: ", len(policy.keys()))


def visualize_td_secret_env_0(nb_iter=10000, max_step=100, gamma=0.999, alpha=0.1, epsilon=0.1):
    env = SecretEnv0()

    sarsa, q_l, exec_time = execute_comparison_td(
        env, nb_iter=nb_iter, alpha=alpha, gamma=gamma, epsilon=epsilon, max_step=max_step)
    visualize_temporal_difference(sarsa, q_l, exec_time, "env0")

def secret_env_0_sarsa(save=False, load=False, run=False, filename="secret_env0_sarsa_policy.json", nb_iter=10000, max_step=100, gamma=0.999, epsilon=0.1, alpha=0.5):
    env = SecretEnv0()

    if not load:
        Q, cummul_avg_sarsa = sarsa(
            env, nb_iter=nb_iter, max_step=max_step, gamma=gamma, epsilon=epsilon, alpha=alpha)
        policy = {}
        for state, actions in Q.items():
            best_action = np.argmax(actions)

            ont_hot_action = np.zeros(env.num_actions())
            ont_hot_action[best_action] = 1.0

            policy[state] = ont_hot_action
    else:
        policy, Q = load_sarsa(filename)

    if save:
        save_sarsa(policy, Q, filename)

    if run:
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


def secret_env_0_q_learning(save=False, load=False, run=False, filename="secret_env_0_policy.json", nb_iter=10000, max_step=100, gamma=0.999, epsilon=0.1, alpha=0.5):
    env = SecretEnv0()

    if not load:
        Q, cummul_avg_q_l = q_learning(
            env, nb_iter=nb_iter, max_step=max_step, gamma=gamma, epsilon=epsilon, alpha=alpha)
        policy = {}
        for state, actions in Q.items():
            best_action = np.argmax(actions)

            ont_hot_action = np.zeros(env.num_actions())
            ont_hot_action[best_action] = 1.0

            policy[state] = ont_hot_action
    else:
        policy, Q = load_sarsa(filename)

    if save:
        save_sarsa(policy, Q, filename)

    if run:
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


def secret_env_0_expected_sarsa(save=False, load=False, run=False, filename="secret_env_0_policy.json", nb_iter=10000, max_step=100, gamma=0.999, epsilon=0.1, alpha=0.5):
    env = SecretEnv0()

    if not load:
        Q, cummul_avg_q_l = expected_sarsa(
            env, nb_iter=nb_iter, max_step=max_step, gamma=gamma, epsilon=epsilon, alpha=alpha)
        policy = {}
        for state, actions in Q.items():
            best_action = np.argmax(actions)

            ont_hot_action = np.zeros(env.num_actions())
            ont_hot_action[best_action] = 1.0

            policy[state] = ont_hot_action
    else:
        policy, Q = load_sarsa(filename)

    if save:
        save_sarsa(policy, Q, filename)

    if run:
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


def secret_env_0_dyna_q(save=False, load=False, run=False, filename="secret_env_0_dyna_q_policy.json", nb_iter=10000, max_step=100, gamma=0.999, epsilon=0.1, alpha=0.1, n=10):
    env = SecretEnv0()

    if not load:
        Q, cummul_avg_q_l = dyna_q(
            env, nb_iter=nb_iter, max_step=max_step, gamma=gamma, epsilon=epsilon, alpha=alpha, planning_steps=n)
        policy = {}
        for state in range(len(Q)):
            best_action = np.argmax(Q[state])

            ont_hot_action = np.zeros(env.num_actions())
            ont_hot_action[best_action] = 1.0

            policy[state] = ont_hot_action
    else:
        policy, Q = load_dyna_q(filename)

    if save:
        save_dyna_q(policy, Q, filename)

    if run:
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


def visualize_mc_secret_env_1(nb_iter=100000, max_step=10000, gamma=0.999):
    env = SecretEnv1()

    es, onp, offp, exec_time = execute_comparison_mc(
        env, nb_iter=nb_iter, max_step=max_step, gamma=gamma)
    visualize_monte_carlo(es, onp, offp, exec_time, "env1")


def visualize_td_secret_env_1(nb_iter=100000, max_step=10000, gamma=0.999, alpha=0.1, epsilon=0.1):
    env = SecretEnv1()

    sarsa, q_l, exec_time = execute_comparison_td(
        env, nb_iter=nb_iter, alpha=alpha, gamma=gamma, epsilon=epsilon, max_step=max_step)
    visualize_temporal_difference(sarsa, q_l, exec_time, "env1")


def secret_env_1_mc_es(save=False, load=False, run=False, display=True, filename="secret_env1_mc_es_policy.json", nb_iter=10000, max_step=100, GAMMA=0.999):
    env = SecretEnv1()

    if not load:
        policy, Q, cummul_avg_es = monte_carlo_with_exploring_start(
            env, nb_iter=nb_iter, max_step=max_step, GAMMA=GAMMA)
    else:
        policy, Q = load_mc_es(filename)

    if save:
        save_mc_es(policy, Q, filename)

    env.reset()

    if run:
        reward = 0
        i = 0
        running = True
        while running:
            action = np.argmax(policy[env.state_id()])
            env.step(action)
            reward += env.score()
            i += 1
            if display:
                env.display()

            if env.is_game_over():
                running = False
                print("Total reward for secret env 1: ", reward)
                print("Number of steps: ", i)
                print("Number of state visited: ", len(policy.keys()))


def secret_env_1_mc_onp(save=False, load=False, run=False, display=True, filename="secret_env1_mc_onp_policy.json", nb_iter=10000, max_step=100, GAMMA=0.999):
    env = SecretEnv1()

    if not load:
        policy, Q, cummul_avg_onp = monte_carlo_on_policy(
            env, nb_iter=nb_iter, max_step=max_step, GAMMA=GAMMA)
    else:
        policy, Q = load_mc_onp(filename)
        print(len(policy.keys()))

    if save:
        save_mc_onp(policy, Q, filename)

    if run:
        reward = 0
        i = 0
        running = run
        while running:
            action = policy[env.state_id()]
            env.step(action)
            reward += env.score()
            i += 1
            if display:
                env.display()

            if env.is_game_over():
                running = False
                print("Total reward for secret env 1: ", reward)
                print("Number of steps: ", i)
                print("Number of state visited: ", len(policy.keys()))


def secret_env_1_mc_offp(save=False, load=False, run=False, display=False, filename="secret_env1_mc_offp_policy.json", nb_iter=10000, max_step=100, GAMMA=0.999):
    env = SecretEnv1()

    if not load:
        policy, Q, cummul_avg_ofp, mean_Q_ofp = monte_carlo_off_policy(
            env, nb_iter=nb_iter, max_step=max_step, gamma=GAMMA)
    else:
        policy, Q = load_mc_onp(filename)

    if save:
        save_mc_offp(policy, Q, filename)

    if run:
        env.reset()
        reward = 0
        i = 0
        running = run
        while running:
            action = np.argmax(policy[env.state_id()])
            env.step(action)
            reward += env.score()
            i += 1
            if display:
                env.display()

            if env.is_game_over():
                running = False
                print("Total reward for secret env 1: ", reward)
                print("Number of steps: ", i)
                print("Number of state visited: ", len(policy.keys()))


def secret_env_1_sarsa(save=False, load=False, run=False, display=False, filename="secret_env1_sarsa_policy.json", nb_iter=10000, max_step=100, gamma=0.999, epsilon=0.1, alpha=0.5):
    env = SecretEnv1()

    if not load:
        Q, cummul_avg_sarsa = sarsa(
            env, nb_iter=nb_iter, max_step=max_step, gamma=gamma, epsilon=epsilon, alpha=alpha)
        policy = {}
        for state, actions in Q.items():
            best_action = np.argmax(actions)

            ont_hot_action = np.zeros(env.num_actions())
            ont_hot_action[best_action] = 1.0

            policy[state] = ont_hot_action
    else:
        policy, Q = load_sarsa(filename)

    if save:
        save_sarsa(policy, Q, filename)

    if run:
        reward = 0
        i = 0
        env.reset()
        running = True

        while running:
            action = np.argmax(policy[env.state_id()])
            env.step(action)
            reward += env.score()
            i += 1
            if display:
                env.display()

            if env.is_game_over():
                running = False
                print("Total reward for secret env 1: ", reward)
                print("Number of steps: ", i)
                print("Number of state visited: ", len(policy.keys()))


def secret_env_1_q_learning(save=False, load=False, run=False, display=False, filename="secret_env_1_policy.json", nb_iter=10000, max_step=100, gamma=0.999, epsilon=0.1, alpha=0.5):
    env = SecretEnv1()

    if not load:
        Q, cummul_avg_q_l = q_learning(
            env, nb_iter=nb_iter, max_step=max_step, gamma=gamma, epsilon=epsilon, alpha=alpha)
        policy = {}
        for state, actions in Q.items():
            best_action = np.argmax(actions)

            ont_hot_action = np.zeros(env.num_actions())
            ont_hot_action[best_action] = 1.0

            policy[state] = ont_hot_action
    else:
        policy, Q = load_sarsa(filename)

    if save:
        save_sarsa(policy, Q, filename)

    if run:
        reward = 0
        i = 0
        env.reset()
        running = True

        while running:
            action = np.argmax(policy[env.state_id()])
            env.step(action)
            reward += env.score()
            i += 1
            if display:
                env.display()

            if env.is_game_over():
                running = False
                print("Total reward for secret env 1: ", reward)
                print("Number of steps: ", i)
                print("Number of state visited: ", len(policy.keys()))


def secret_env_1_dyna_q(save=False, load=False, run=False, display=False, filename="secret_env_1_dyna_q_policy.json", nb_iter=10000, max_step=100, gamma=0.999, epsilon=0.1, alpha=0.1, n=10):
    env = SecretEnv1()

    if not load:
        Q, cummul_avg_q_l = dyna_q(
            env, nb_iter=nb_iter, max_step=max_step, gamma=gamma, epsilon=epsilon, alpha=alpha, planning_steps=n)
        policy = {}
        for state in range(len(Q)):
            best_action = np.argmax(Q[state])

            ont_hot_action = np.zeros(env.num_actions())
            ont_hot_action[best_action] = 1.0

            policy[state] = ont_hot_action
    else:
        policy, Q = load_dyna_q(filename)

    if save:
        save_dyna_q(policy, Q, filename)

    if run:
        reward = 0
        i = 0
        env.reset()
        running = True

        while running:
            action = np.argmax(policy[env.state_id()])
            env.step(action)
            reward += env.score()
            i += 1
            if display:
                env.display()

            if env.is_game_over():
                running = False
                print("Total reward for secret env 1: ", reward)
                print("Number of steps: ", i)
                print("Number of state visited: ", len(policy.keys()))


def visualize_mc_secret_env_2(nb_iter=100000, max_step=10000, gamma=0.999):
    env = SecretEnv2()

    es, onp, offp, exec_time = execute_comparison_mc(
        env, nb_iter=nb_iter, max_step=max_step, gamma=gamma)
    visualize_monte_carlo(es, onp, offp, exec_time, "env2")


def visualize_td_secret_env_2(nb_iter=100000, max_step=10000, gamma=0.999, alpha=0.1, epsilon=0.1):
    env = SecretEnv2()

    sarsa, q_l, exec_time = execute_comparison_td(
        env, nb_iter=nb_iter, alpha=alpha, gamma=gamma, epsilon=epsilon, max_step=max_step)
    visualize_temporal_difference(sarsa, q_l, exec_time, "env2")


def visualize_mc_secret_env_3(nb_iter=100000, max_step=10000, gamma=0.999):
    env = SecretEnv3()

    es, onp, offp, exec_time = execute_comparison_mc(
        env, nb_iter=nb_iter, max_step=max_step, gamma=gamma)
    visualize_monte_carlo(es, onp, offp, exec_time, "env3")


def visualize_td_secret_env_3(nb_iter=100000, max_step=10000, gamma=0.999, alpha=0.1, epsilon=0.1):
    env = SecretEnv3()

    sarsa, q_l, exec_time = execute_comparison_td(
        env, nb_iter=nb_iter, alpha=alpha, gamma=gamma, epsilon=epsilon, max_step=max_step)
    visualize_temporal_difference(sarsa, q_l, exec_time, "env3")

import pygame
import numpy as np
import pprint
import time
import matplotlib.pyplot as plt

from utils import *
from test import *
from secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3
from environments import LineWorld, GridWorld
from algorithms import value_iteration, policy_iteration, monte_carlo_with_exploring_start, monte_carlo_on_policy, monte_carlo_off_policy, sarsa, expected_sarsa, q_learning, dyna_q


def line_world():
    # line_world_vi(save=False, load=True, run=True)
    # line_world_pi(save=True, load=False, run=False)
    line_world_mc_es(save=False, load=True, run=True)
    # line_world_mc_onp()
    # line_world_mc_offp()
    # line_world_sarsa()

def grid_world():
    env = GridWorld(width=5, height=5)

    start = time.time()
    # policy, V, episode = value_iteration(env)
    # policy, V, episode = policy_iteration(env, GAMMA=0.3)
    # policy, Q = monte_carlo_with_exploring_start(
    #     env, nb_iter=10000, max_step=100, GAMMA=0.999)
    # policy, Q = monte_carlo_on_policy(env, nb_iter=10000)
    # policy, Q = monte_carlo_off_policy(env, nb_iter=10000)
    Q = sarsa(env, nb_iter=10000)
    print("Time taken to train: ", time.time() - start)

    policy = {}
    for state, actions in Q.items():
        best_action = np.argmax(actions)

        ont_hot_action = np.zeros(env.num_actions())
        ont_hot_action[best_action] = 1.0

        policy[state] = ont_hot_action

    V = Q

    if isinstance(policy, dict):
        print("Optimal policy: \n")
        pprint.pprint(policy)

        print(V)
    else:
        print("Optimal policy: \n", policy)
        print("", V)

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

        action = np.argmax(policy[env.state_id()])
        print(env.available_actions())
        env.step(action)
        env.render(screen)
        clock.tick(5)

        if env.is_game_over():
            return
    
    pygame.quit()


def secret_env0():

    secret_env_0_sarsa()
    # Approx. 8k states
    # env = SecretEnv0()

    # exec_time = {}

    # policy, V, episode = value_iteration(env)
    # policy, V, episode = policy_iteration(env)
    # policy, Q = monte_carlo_with_exploring_start(
    #     env, nb_iter=10000, max_step=8000)
    # policy, Q, cummul_avg_onp = monte_carlo_on_policy(
    #     env, nb_iter=10000, max_step=8000)
    # policy, Q, cummul, mean_q_avg = monte_carlo_off_policy(
    #     env, nb_iter=10000, max_step=1000)

    # execute_comparison_mc(env, nb_iter=)

    # Q_sarsa, cumm_avg_sarsa, Q_q_l, cumm_avg_q_l, Q_expected_sarsa, cumm_avg_sarsa_e, exec_time = execute_comparison_td(
    #     env, nb_iter=50000, alpha=0.3, gamma=0.999, epsilon=0.1, max_step=2500)
    # visualize_temporal_difference(
    #     cumm_avg_sarsa, cumm_avg_q_l, cumm_avg_sarsa_e, exec_time, "env0")

    # policy = {}
    # for state, actions in Q.items():
    #     best_action = np.argmax(actions)

    #     ont_hot_action = np.zeros(env.num_actions())
    #     ont_hot_action[best_action] = 1.0

    #     policy[state] = ont_hot_action

    # V = Q

    # env.reset()

    # reward = 0
    # i = 0
    # running = True
    # while running:
    #     action = policy[env.state_id()]
    #     env.step(action)
    #     reward += env.score()
    #     i += 1
    #     env.display()

    #     if env.is_game_over():
    #         env.reset()
    #         print("Total reward for secret env 0: ", reward)
    #         print("Number of steps: ", i)
    #         print("Number of state visited: ", len(policy.keys()))
    #         running = False


def secret_env1():
    # 65 536 states
    env = SecretEnv1()
    start = time.time()
    # policy, V, episode = value_iteration(env)
    # policy, V, episode = policy_iteration(env)
    # policy, Q = monte_carlo_with_exploring_start(
    #     env, nb_iter=100000, max_step=1000)
    policy, Q, cummul_avg_onp = monte_carlo_on_policy(
        env, nb_iter=10000, max_step=8000)
    # policy, Q = monte_carlo_off_policy(env, nb_iter=100000, max_step=10000)

    # Q_sarsa, cumm_avg_sarsa, Q_q_l, cumm_avg_q_l, Q_expected_sarsa, cumm_avg_sarsa_e, exec_time = execute_comparison_td(
    #     env, nb_iter=15000, alpha=0.1, gamma=0.999, epsilon=0.1, max_step=2500)
    # visualize_temporal_difference(
    #     cumm_avg_sarsa, cumm_avg_q_l, cumm_avg_sarsa_e, exec_time, "env1")

    # policy = {}
    # for state, actions in Q.items():
    #     best_action = np.argmax(actions)

    #     ont_hot_action = np.zeros(env.num_actions())
    #     ont_hot_action[best_action] = 1.0

    #     policy[state] = ont_hot_action

    env.reset()

    reward = 0
    i = 0
    running = True
    while running:
        action = policy[env.state_id()]
        env.step(action)
        reward += env.score()
        env.display()
        i += 1

        if env.is_game_over():
            env.reset()
            print("Total reward for secret env 1: ", reward)
            print("Number of steps: ", i)
            print("Number of state visited: ", len(policy.keys()))
            running = False


def secret_env2():
    # 2 097 152
    env = SecretEnv2()

    start = time.time()
    # policy, V, episode = value_iteration(env)
    # policy, V, episode = policy_iteration(env)
    # policy, Q = monte_carlo_with_exploring_start(nv, nb_iter=100000, max_step=10000)
    policy, Q, cummul_avg_onp = monte_carlo_on_policy(
        env, nb_iter=100000, max_step=8000)
    # policy, Q = monte_carlo_off_policy(env, nb_iter=100000, max_step=10000)

    # Q_sarsa, cumm_avg_sarsa, Q_q_l, cumm_avg_q_l, Q_expected_sarsa, cumm_avg_sarsa_e, exec_time = execute_comparison_td(
    #     env, nb_iter=15000, alpha=0.1, gamma=0.999, epsilon=0.1, max_step=2500)
    # visualize_temporal_difference(
    #     cumm_avg_sarsa, cumm_avg_q_l, cumm_avg_sarsa_e, exec_time, "env2")

    # policy = {}

    # for state, actions in Q.items():
    #     best_action = np.argmax(actions)

    #     ont_hot_action = np.zeros(env.num_actions())
    #     ont_hot_action[best_action] = 1.0

    #     policy[state] = ont_hot_action

    env.reset()

    reward = 0
    i = 0
    running = True
    while running:
        action = np.argmax(policy[env.state_id()])
        env.step(action)
        reward += env.score()
        env.display()
        i += 1

        if env.is_game_over():
            env.reset()
            print("Total reward for secret env 2: ", reward)
            print("Number of steps: ", i)
            print("Number of state visited: ", len(policy.keys()))
            running = False


def secret_env3():
    # 65 536
    env = SecretEnv3()
    start = time.time()
    # policy, V, episode = value_iteration(env)
    # policy, V, episode = policy_iteration(env)
    # policy, Q = monte_carlo_with_exploring_start( env, nb_iter=100000, max_step=25000)
    policy, Q, cummul_avg_onp = monte_carlo_on_policy(
        env, nb_iter=10000, max_step=8000)
    # policy, Q = monte_carlo_off_policy(env, nb_iter=10000, max_steps=1000)
    print("Time taken to train: ", time.time() - start)

    # if isinstance(policy, dict):
    #     print("Optimal policy: \n")
    #     pprint.pprint(policy)
    # else:
    #     print("Optimal policy: \n", policy)

    env.reset()

    reward = 0
    running = True
    while running:
        action = np.argmax(policy[env.state_id()])
        env.step(action)
        reward += env.score()
        env.display()

        if env.is_game_over():
            env.reset()
            print("Total reward: ", reward)
            running = False


if __name__ == "__main__":
    line_world()
    # grid_world()
    # secret_env0()
    # secret_env1()
    # secret_env2()
    # secret_env3()

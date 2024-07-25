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
    filename = ""
    # line_world_vi(save=False, load=True, run=True)
    # line_world_pi(save=True, load=False, run=False)
    # line_world_mc_es(save=False, load=True, run=True)
    # line_world_mc_onp(save=True, load=False, run=True)
    # line_world_mc_offp(save=False, load=True, run=True)
    # line_world_sarsa(save=False, load=False, run=True)
    # line_world_q_learning(save=True, load=False, run=True, nb_iter=10000,
    #                       alpha=0.1, gamma=0.999, epsilon=0.1, max_step=1000)
    # line_world_dyna_q(save=False, load=False, run=True)
    # visualize_mc_line_world(nb_iter=10000, max_step=100, gamma=0.999)
    visualize_td_line_world()



def secret_env0():
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.999
    max_step = 1000
    nb_iter = 10000

    filename = "secret_env_0_q_learning_{}_{}_{}_{}_{}_{}".format(
        nb_iter, alpha, gamma, epsilon, max_step, time.time())
    # secret_env_0_mc_es(save=True, load=False, run=True)
    # secret_env_0_mc_onp(save=False, load=True, run=True)
    # secret_env_0_mc_offp(save=False, load=False, run=True,
    #                      nb_iter=10000, max_step=1000)
    # secret_env_0_sarsa(save=True, load=False, run=True)
    # secret_env_0_expected_sarsa(save=True,
    #                             load=False,
    #                             run=True,
    #                             nb_iter=nb_iter,
    #                             alpha=alpha,
    #                             gamma=gamma,
    #                             epsilon=epsilon
    #                             )
    # secret_env_0_q_learning(save=False,
    #                         load=False,
    #                         run=True,
    #                         # filename=filename,
    #                         nb_iter=nb_iter,
    #                         alpha=alpha,
    #                         gamma=gamma,
    #                         epsilon=epsilon,
    #                         max_step=max_step
    #                         )

    # secret_env_0_dyna_q(save=True, load=False, run=True)

    # visualize_mc_secret_env_0()
    # visualize_td_secret_env_0(nb_iter=50000, max_step=8000)





def secret_env1():
    # 65 536 states
    env = SecretEnv1()

    # visualize_mc_secret_env_1(nb_iter=50000, max_step=10000, gamma=0.999)
    # visualize_td_secret_env_1(nb_iter=50000, max_step=10000)
    # secret_env_1_mc_es(save=False, load=False, run=False, display=False,
    #                    nb_iter=50000, max_step=10000)
    # secret_env_1_mc_onp(save=False, load=False, run=False, display=False,
    #                     nb_iter=10000, max_step=100000, GAMMA=0.1)
    # secret_env_1_mc_offp(save=False, load=False, run=False, display=False,
    #                      nb_iter=50000, max_step=10000)

    # secret_env_1_sarsa(save=True, load=False, run=True, display=False,
    #                    nb_iter=50000, max_step=10000)
    # secret_env_1_q_learning(save=True, load=False, run=True, display=False,
    #                         nb_iter=50000, max_step=10000)

    secret_env_1_dyna_q(save=True, load=False, run=True, display=False,
                        nb_iter=10000, max_step=1000)



def secret_env2():
    # 2 097 152
    env = SecretEnv2()
    env.reset()

    start = time.time()



def secret_env3():
    # 65 536
    env = SecretEnv3()

    visualize_mc_secret_env_3(nb_iter=50000, max_step=10000)
    visualize_td_secret_env_3(nb_iter=50000, max_step=10000)


if __name__ == "__main__":
    # line_world()
    # grid_world()
    # secret_env0()
    secret_env1()
    # secret_env2()
    # secret_env3()

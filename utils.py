import time
import numpy as np
import matplotlib.pyplot as plt
from algorithms import monte_carlo_with_exploring_start, monte_carlo_on_policy, monte_carlo_off_policy, sarsa, expected_sarsa, q_learning


def execute_comparison_mc(env, nb_iter, max_step, gamma):
    exec_time = {}

    start = time.time()
    _, _, cummu_avg_r_es, mean_q_value_es = monte_carlo_with_exploring_start(
        env, nb_iter=nb_iter, max_step=max_step, gamma=gamma)
    exec_time['exploring_start'] = time.time() - start

    start = time.time()
    _, _, cummu_avg_r_onp, mean_q_value_onp = monte_carlo_on_policy(
        env, nb_iter=nb_iter, max_step=max_step, gamma=gamma)
    exec_time['on_policy'] = time.time() - start

    start = time.time()
    _, _, cummu_avg_r_ofp, mean_q_value_ofp = monte_carlo_off_policy(
        env, nb_iter=nb_iter, max_step=max_step, gamma=gamma)
    exec_time['off_policy'] = time.time() - start

    return cummu_avg_r_es, cummu_avg_r_onp, cummu_avg_r_ofp, exec_time

def execute_comparison_td(env, nb_iter, alpha, gamma, epsilon, max_step):
    exec_time = {}

    start = time.time()
    Q_sarsa, cumm_avg_sarsa = sarsa(
        env, nb_iter=nb_iter, alpha=alpha, max_step=max_step, gamma=gamma, epsilon=epsilon)
    exec_time['sarsa'] = time.time() - start
    start = time.time()
    Q_q_l, cumm_avg_q_l = q_learning(
        env, nb_iter=nb_iter, alpha=alpha, max_step=max_step, gamma=gamma, epsilon=epsilon)
    exec_time['q_learning'] = time.time() - start
    start = time.time()
    Q_expected_sarsa, cumm_avg_sarsa_e = expected_sarsa(
        env, nb_iter=nb_iter, alpha=alpha, max_step=max_step, gamma=gamma, epsilon=epsilon)
    exec_time['extected_sarsa'] = time.time() - start

    return cumm_avg_sarsa, cumm_avg_q_l, cumm_avg_sarsa_e, exec_time


def visualize_monte_carlo(exploring_start, on_policy, off_policy, execution_time, env_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(np.arange(len(exploring_start)),
             exploring_start, label='ExploringStart', color='blue')
    ax1.plot(np.arange(len(on_policy)), on_policy,
             label='On Policy', color='red')
    ax1.plot(np.arange(len(off_policy)), off_policy,
             label='Off Policy', color='green')
    ax1.set_title('Cumulative Reward of MC ES vs On Policy vs Off Policy')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Cumulative Reward')
    ax1.legend()
    ax1.grid(True)

    algorithms = list(execution_time.keys())
    times = list(execution_time.values())
    ax2.bar(algorithms, times, color=['blue', 'red', 'green'])
    ax2.set_title('Execution Time of MC ES vs On Policy vs Off Policy')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Time (seconds)')

    fig.suptitle(
        f'Comparison of MC learning methods on {env_name}', fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_temporal_difference(sarsa, expected_sarsa, q_learning, execution_time, env_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(np.arange(len(sarsa)),
             sarsa, label='SARSA', color='blue')
    ax1.plot(np.arange(len(expected_sarsa)), expected_sarsa,
             label='Expected SARSA', color='red')
    ax1.plot(np.arange(len(q_learning)), q_learning,
             label='Q-Learning', color='green')
    ax1.set_title(
        'Cumulative Reward of TD SARSA vs Expected SARSA vs Q-Learning')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Cumulative Reward')
    ax1.legend()
    ax1.grid(True)

    algorithms = list(execution_time.keys())
    times = list(execution_time.values())
    ax2.bar(algorithms, times, color=['blue', 'red', 'green'])
    ax2.set_title('Execution Time of TD SARSA vs Expected SARSA vs Q-Learning')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Time (seconds)')

    fig.suptitle(
        f'Comparison of TD learning methods on {env_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

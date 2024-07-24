import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from algorithms import monte_carlo_with_exploring_start, monte_carlo_on_policy, monte_carlo_off_policy, sarsa, expected_sarsa, q_learning

OUTPUT_DIR = 'output/'

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


def save_policy_dynamic(policy, value_function, filename):

    policy = {key: int(value) for key, value in policy.items()}
    value_function = value_function.tolist()

    try:
        with open(OUTPUT_DIR+filename, 'w') as file:
            json.dump(policy, file)
        print(f"Policy saved successfully to {filename}.")
    except IOError as e:
        print(f"An error occurred while saving the policy: {e}")

    try:
        with open(OUTPUT_DIR+filename.replace('policy', 'value'), 'w') as file:
            json.dump(value_function, file)
        print(
            f"Value function saved successfully to {filename.replace('policy', 'value')}.")
    except IOError as e:
        print(f"An error occurred while saving the value function: {e}")


def load_policy_dynamic(filename):
    try:
        with open(OUTPUT_DIR+filename, 'r') as file:
            policy = json.load(file)
            policy = {int(key): value for key, value in policy.items()}
            print(f"Policy loaded successfully from {filename}.")
    except IOError as e:
        print(f"An error occurred while loading the policy: {e}")
        return None

    try:
        with open(OUTPUT_DIR+filename.replace('policy', 'value'), 'r') as file:
            value_function = json.load(file)
            print(
                f"Value function loaded successfully from {filename.replace('policy', 'value')}.")
    except IOError as e:
        print(f"An error occurred while loading the value function: {e}")
        return None

    return np.array(value_function), policy


def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d


def save_mc_es(policy, Q, policy_filename):
    policy_dict = convert_defaultdict_to_dict(policy)
    Q_dict = convert_defaultdict_to_dict(Q)

    try:
        with open(OUTPUT_DIR+policy_filename, 'w') as policy_file:
            json.dump(policy_dict, policy_file)
    except IOError as e:
        print(f"An error occurred while saving the policy: {e}")

    try:
        with open(OUTPUT_DIR+policy_filename.replace('policy', 'Q'), 'w') as Q_file:
            json.dump(Q_dict, Q_file)
    except IOError as e:
        print(f"An error occurred while saving the Q function: {e}")


def load_mc_es(policy_filename):
    try:
        with open(OUTPUT_DIR+policy_filename, 'r') as policy_file:
            policy_dict = json.load(policy_file)
            print(f"Policy loaded successfully from {policy_filename}.")
    except IOError as e:
        print(f"An error occurred while loading the policy: {e}")
        return None

    try:
        with open(OUTPUT_DIR+policy_filename.replace('policy', 'Q'), 'r') as Q_file:
            Q_dict = json.load(Q_file)
            print(
                f"Q function loaded successfully from {policy_filename.replace('policy', 'Q')}.")
    except IOError as e:
        print(f"An error occurred while loading the Q function: {e}")
        return None

    policy = defaultdict(lambda: [0.0, 1.0], policy_dict)
    Q = defaultdict(lambda: [0.0, 0.0], Q_dict)

    return policy, Q


def save_mc_onp(policy, Q, filename):
    Q_dict = {
        str(key): float(value)
        for key, value in Q.items()
    }
    try:
        with open(OUTPUT_DIR+filename, 'w') as file:
            json.dump(policy, file)
            print(f"Policy saved successfully to {filename}.")
    except IOError as e:
        print(f"An error occurred while saving the policy: {e}")

    try:
        with open(OUTPUT_DIR+filename.replace('policy', 'Q'), 'w') as file:
            json.dump(Q_dict, file)
            print(
                f"Q function saved successfully to {filename.replace('policy', 'Q')}.")
    except IOError as e:
        print(f"An error occurred while saving the Q function: {e}")


def load_mc_onp(filename):
    try:
        with open(OUTPUT_DIR+filename, 'r') as file:
            policy = json.load(file)
            policy = {int(key): value for key,
                      value in policy.items() if key != 'null'}
            print(f"Policy loaded successfully from {filename}.")
    except IOError as e:
        print(f"An error occurred while loading the policy: {e}")
        return None

    try:
        with open(OUTPUT_DIR+filename.replace('policy', 'Q'), 'r') as file:
            Q = json.load(file)
            print(
                f"Q function loaded successfully from {filename.replace('policy', 'Q')}.")
    except IOError as e:
        print(f"An error occurred while loading the Q function: {e}")
        return None

    return policy, Q


def save_mc_offp(policy, Q, filename):
    Q_dict = {int(k): v.tolist() for k, v in Q.items()}
    try:
        with open(OUTPUT_DIR+filename, 'w') as file:
            json.dump(policy, file)
            print(f"Policy saved successfully to {filename}.")
    except IOError as e:
        print(f"An error occurred while saving the policy: {e}")

    try:
        with open(OUTPUT_DIR+filename.replace('policy', 'Q'), 'w') as file:
            json.dump(Q_dict, file)
            print(
                f"Q function saved successfully to {filename.replace('policy', 'Q')}.")
    except IOError as e:
        print(f"An error occurred while saving the Q function: {e}")


def save_sarsa(policy, Q, filename):
    Q_dict = {str(k): v for k, v in Q.items()}
    policy = {int(k): v.tolist() for k, v in policy.items()}
    try:
        with open(OUTPUT_DIR+filename, 'w') as file:
            json.dump(policy, file)
            print(f"Policy saved successfully to {filename}.")
    except IOError as e:
        print(f"An error occurred while saving the policy: {e}")

    try:
        with open(OUTPUT_DIR+filename.replace('policy', 'Q'), 'w') as file:
            json.dump(Q_dict, file)
            print(
                f"Q function saved successfully to {filename.replace('policy', 'Q')}.")
    except IOError as e:
        print(f"An error occurred while saving the Q function: {e}")


def load_sarsa(filename):
    try:
        with open(OUTPUT_DIR+filename, 'r') as file:
            policy = json.load(file)
            policy = {int(key): np.array(value)
                      for key, value in policy.items()}
            print(f"Policy loaded successfully from {filename}.")
    except IOError as e:
        print(f"An error occurred while loading the policy: {e}")
        return None

    try:
        with open(OUTPUT_DIR+filename.replace('policy', 'Q'), 'r') as file:
            Q = json.load(file)
            Q = {tuple(map(int, k.split(','))): np.array(v)
                 for k, v in Q.items()}
            print(
                f"Q function loaded successfully from {filename.replace('policy', 'Q')}.")
    except IOError as e:
        print(f"An error occurred while loading the Q function: {e}")
        return None

    return policy, Q


def save_dyna_q(policy, Q, filename):
    Q_list = Q.tolist()

    policy = {int(k): v.tolist() for k, v in policy.items()}

    try:
        with open(OUTPUT_DIR+filename, 'w') as file:
            json.dump(policy, file)
            print(f"Policy saved successfully to {filename}.")
    except IOError as e:
        print(f"An error occurred while saving the policy: {e}")

    try:
        with open(OUTPUT_DIR+filename.replace('policy', 'Q'), 'w') as file:
            json.dump(Q_list, file)
            print(
                f"Q function saved successfully to {filename.replace('policy', 'Q')}.")
    except IOError as e:
        print(f"An error occurred while saving the Q function: {e}")


def load_dyna_q(filename):
    try:
        with open(OUTPUT_DIR+filename, 'r') as file:
            policy = json.load(file)
            policy = {int(key): np.array(value)
                      for key, value in policy.items()}
            print(f"Policy loaded successfully from {filename}.")
    except IOError as e:
        print(f"An error occurred while loading the policy: {e}")
        return None

    try:
        with open(OUTPUT_DIR+filename.replace('policy', 'Q'), 'r') as file:
            Q = json.load(file)
            Q = np.array(Q)
            print(
                f"Q function loaded successfully from {filename.replace('policy', 'Q')}.")
    except IOError as e:
        print(f"An error occurred while loading the Q function: {e}")
        return None

    return policy, Q

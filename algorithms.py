import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple
from environments import Environment


def value_iteration(env, GAMMA: float = 0.999, THETA: float = 0.00001):
    num_states = env.num_states()
    num_actions = env.num_actions()
    
    V = np.zeros(num_states)
    
    policy = {}

    episode = 1
    
    while True:
        delta = 0
        V_copy = np.copy(V)
        
        for s in range(num_states):
            action_values = np.zeros(num_actions)
            for a in range(num_actions):
                action_value = 0
                for s_prime in range(num_states):
                    for r_index in range(env.num_rewards()):
                        prob = env.p(s, a, s_prime, r_index)
                        reward = env.reward(r_index)
                        action_value += prob * (reward + GAMMA * V_copy[s_prime])
                action_values[a] = action_value
            
            V[s] = np.max(action_values)
            delta = max(delta, abs(V[s] - V_copy[s]))
        
        if delta < THETA:
            break
        episode += 1
    
    for s in range(num_states):
        action_values = np.zeros(num_actions)
        for a in range(num_actions):
            action_value = 0
            for s_prime in range(num_states):
                for r_index in range(env.num_rewards()):
                    prob = env.p(s, a, s_prime, r_index)
                    reward = env.reward(r_index)
                    action_value += prob * (reward + GAMMA * V[s_prime])
            action_values[a] = action_value
        policy[s] = np.argmax(action_values)
    
    print("Value Iteration episodes = ", episode)
    return V, policy


def policy_iteration(env, GAMMA: float = 0.999, THETA: float = 0.0001):
    num_states = env.num_states()
    num_actions = env.num_actions()

    policy = np.zeros(num_states, dtype=int)
    
    V = np.zeros(num_states)
    
    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(num_states):
                v = V[s]
                action = policy[s]
                V[s] = sum(env.p(s, action, s_prime, r_index) * (env.reward(r_index) + GAMMA * V[s_prime])
                           for s_prime in range(num_states)
                           for r_index in range(env.num_rewards()))
                delta = max(delta, abs(v - V[s]))
            if delta < THETA:
                break
        
        # Policy Improvement
        policy_stable = True
        for s in range(num_states):
            old_action = policy[s]
            action_values = np.zeros(num_actions)
            for a in range(num_actions):
                action_values[a] = sum(env.p(s, a, s_prime, r_index) * (env.reward(r_index) + GAMMA * V[s_prime])
                                      for s_prime in range(num_states)
                                      for r_index in range(env.num_rewards()))
            policy[s] = np.argmax(action_values)
            
            if old_action != policy[s]:
                policy_stable = False
        
        if policy_stable:
            break
    
    target_policy = {}
    for i in range(len(policy)):
        target_policy[i] = policy[i]
    
    return V, target_policy


def monte_carlo_with_exploring_start(
        env: Environment,
        nb_iter: int = 10000,
        GAMMA: float = 0.999,
        max_step: int = 100
) -> Tuple[Dict[Tuple, List[float]], Dict[Tuple, List[float]]]:

    num_actions = env.num_actions()

    Q: Dict[Tuple, List[float]] = defaultdict(lambda: [0.0] * num_actions)
    pi: Dict[Tuple, List[float]] = defaultdict(
        lambda: [1.0 / num_actions] * num_actions)
    Returns: Dict[Tuple, List[float]] = defaultdict(list)

    cummulative_reward_avg = []
    mean_Q_value = []

    for it in tqdm(range(nb_iter)):
        env = env.from_random_state()
        is_first_action = True
        trajectory = []
        steps_count = 0
        episode_reward = 0

        while not env.is_game_over() and steps_count < max_step:
            s = env.state_id()
            aa = env.available_actions()

            if s not in pi or all(p == 0 for p in pi[s]):
                pi[s] = [
                    1.0 / len(aa) if a in aa else 0.0 for a in range(num_actions)]

            if is_first_action:
                a = np.random.choice(aa)
                is_first_action = False
            else:
                a = np.argmax(pi[s])

            while env.is_forbidden(a):
                allowed_action = [a for a in aa if not env.is_forbidden(a)]
                a = np.random.choice(allowed_action)

            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score

            trajectory.append((s, a, r, aa))
            steps_count += 1
            episode_reward += r

        cummulative_reward_avg.append(episode_reward)

        G = 0
        for t, (s, a, r, aa) in reversed(list(enumerate(trajectory))):
            G = GAMMA * G + r

            if all(triplet[0] != s or triplet[1] != a for triplet in trajectory[:t]):
                Returns[(s, a)].append(G)
                Q[s][a] = np.mean(Returns[(s, a)])

                best_a = None
                best_a_score = -float('inf')
                for action in aa:
                    if (s, action) not in Q:
                        Q[s][action] = np.random.random()
                    if Q[s][action] > best_a_score:
                        best_a = action
                        best_a_score = Q[s][action]

                pi[s] = [1.0 if i ==
                         best_a else 0.0 for i in range(num_actions)]
                for action in range(num_actions):
                    if action not in aa:
                        pi[s][action] = 0.0

        total_q_value = sum(sum(q) for q in Q.values())
        num_state_action_pairs = sum(len(q) for q in Q.values())
        mean_q_value = total_q_value / num_state_action_pairs
        mean_Q_value.append(mean_q_value)

    return pi, Q, cummulative_reward_avg, mean_Q_value


def epsilon_greedy_policy(Q, epsilon, state, available_actions, n_action):
    A = np.zeros(n_action, dtype=float)
    n_available_actions = len(available_actions)

    if n_available_actions == 0:
        return A

    if random.uniform(0, 1) < epsilon:
        prob = 1.0 / len(available_actions)
        for action in available_actions:
            A[action] = prob
    else:
        best_action = max(available_actions, key=lambda a: Q[(state, a)])
        A[best_action] = 1.0
    return A

def monte_carlo_on_policy(
        env: Environment,
        nb_iter: int = 10000,
        GAMMA: float = 0.999,
        max_step: int = 100,
        epsilon: float = 0.1
) -> Tuple[Dict[Tuple, List[float]], Dict[Tuple, List[float]]]:

    num_actions = env.num_actions()

    Q = defaultdict(float)
    total_return = defaultdict(float)
    N = defaultdict(int)

    cummulative_reward_avg = []

    for it in tqdm(range(nb_iter)):
        s = env.reset()
        trajectory = []
        steps_count = 0
        episode_reward = 0

        while not env.is_game_over() and steps_count < max_step:
            aa = env.available_actions()
            action_probs = epsilon_greedy_policy(
                Q, epsilon, s, aa, num_actions)
            a = np.random.choice(aa, p=[action_probs[a] for a in aa])

            env.step(a)
            r = env.score()
            episode_reward += r

            trajectory.append((s, a, r, aa))
            steps_count += 1

            s = env.state_id()

        cummulative_reward_avg.append(episode_reward)

        all_state_actions = [(s, a) for s, a, _, _ in trajectory]
        rewards = [r for _, _, r, _ in trajectory]

        for t, (s, a, r, aa) in enumerate(trajectory):

            if not (s, a) in all_state_actions[0:t]:
                R = GAMMA * sum(rewards[t:])
                total_return[(s, a)] += R

                N[(s, a)] += 1

                Q[(s, a)] = total_return[(s, a)] / N[(s, a)]

    policy = {}
    for (s, a), q in Q.items():
        if s not in policy or Q[(s, policy[s])] < q:
            policy[s] = a

    if None in policy:
        policy[0] = policy[None]

    return policy, Q, cummulative_reward_avg


def get_random_policy(env):
    num_actions = env.num_actions()
    available_actions = env.available_actions()
    state_policy = {}

    state_policy = {action: 0.0 for action in range(num_actions)}
    action_prob = 1.0 / len(available_actions) if available_actions else 0.0

    for action in available_actions:
        state_policy[action] = action_prob

    return state_policy

def monte_carlo_off_policy(
    env,
    gamma: float = 0.999,
    nb_iter: int = 10000,
    max_step: int = 10,
    epsilon: float = 0.1
) -> Tuple[Dict[Tuple, List[float]], Dict[Tuple, List[float]]]:

    num_state = env.num_states()
    num_actions = env.num_actions()

    Q = defaultdict(lambda: np.zeros(num_actions))
    C = defaultdict(lambda: np.zeros(num_actions))
    target_policy = {}
    b = {}

    cummulative_reward_avg = []
    mean_Q_value = []

    for _ in tqdm(range(nb_iter)):
        env.reset()
        trajectory = []
        episode_reward = 0
        step_count = 0

        s = env.state_id()

        if s not in b:
            b[s] = get_random_policy(env)

        while not env.is_game_over() and step_count < max_step:
            action_probs = b[s]
            valid_actions = [
                a for a, probs in action_probs.items() if probs > 0]

            if valid_actions:
                action = np.random.choice(valid_actions)
            else:
                action = np.random.choice(env.available_actions())

            env.step(action)
            r = env.score()
            trajectory.append((s, action, r, b[s]))
            episode_reward += r
            s = env.state_id()
            step_count += 1

            if s not in b:
                b[s] = get_random_policy(env)

        cummulative_reward_avg.append(episode_reward)

        G = 0
        W = 1

        # Loop for each step of the episode in reverse order
        for t in reversed(range(len(trajectory))):
            s, a, r, aa = trajectory[t]
            G = gamma * G + r

            C[s][a] += W
            Q[s][a] += (W/C[s][a]) * (G - Q[s][a])

            if s not in target_policy:
                target_policy[s] = {
                    a: 1/num_actions for a in range(num_actions)}

            best_action = np.argmax(Q[s])
            for a in range(num_actions):
                target_policy[s][a] = epsilon / num_actions + \
                    (1 - epsilon) * (1 if a == best_action else 0)

            W *= 1 / b[s][a]

        mean_Q_value.append(np.mean([Q[s][a]
                            for s in Q for a in range(num_actions)]))

    return target_policy, Q, cummulative_reward_avg, mean_Q_value


def sarsa_epsilon_greedy_policy(Q, epsilon, state, available_actions, n_action):
    A = np.zeros(n_action, dtype=float)
    n_available_actions = len(available_actions)

    if n_available_actions == 0:
        return A

    equal_prob = epsilon / n_available_actions

    for action in available_actions:
        A[action] = equal_prob

    best_action = available_actions[np.argmax(
        [Q[state][a] for a in available_actions])]
    A[best_action] += 1 - epsilon

    assert np.isclose(np.sum(A), 1.0)

    return A


def sarsa(env,
          nb_iter: int = 10000,
          epsilon: float = 0.1,
          alpha: float = 0.5,
          gamma: float = 0.999,
          max_step: int = 1000
          ):
    num_actions = env.num_actions()

    Q = defaultdict(lambda: [0.0] * num_actions)
    cummulative_reward_avg = []

    for it in tqdm(range(nb_iter)):
        env.reset()
        state = env.state_id()
        aa = env.available_actions()

        action_probs = sarsa_epsilon_greedy_policy(
            Q, epsilon, state, aa, num_actions)
        action = np.random.choice(aa, p=[action_probs[a] for a in aa])

        num_step = 0
        total_reward = 0.
        while not env.is_game_over() and num_step < max_step:
            env.step(action)
            next_state = env.state_id()
            reward = env.score()

            aa = env.available_actions()
            next_action_probs = sarsa_epsilon_greedy_policy(
                Q, epsilon, next_state, aa, num_actions)
            next_action = np.random.choice(
                aa, p=[next_action_probs[a] for a in aa])

            Q[state][action] += alpha * \
                (reward + gamma * Q[next_state]
                 [next_action] - Q[state][action])

            state = next_state
            action = next_action

            num_step += 1
            total_reward += reward
        cummulative_reward_avg.append(total_reward)

    return Q, cummulative_reward_avg


def expected_sarsa(env,
                     nb_iter: int = 10000,
                     epsilon: float = 0.1,
                     alpha: float = 0.5,
                     gamma: float = 0.999,
                     max_step: int = 1000
):
    num_actions = env.num_actions()
    
    Q = defaultdict(lambda: [0.0] * num_actions)
    cummulative_reward_avg = []
    
    for it in tqdm(range(nb_iter)):
        env.reset()
        state = env.state_id()
        total_reward = 0.
        step = 0

        aa = env.available_actions()
        action_probs = sarsa_epsilon_greedy_policy(
            Q, epsilon, state, aa, num_actions)
        action = np.random.choice(aa, p=[action_probs[a] for a in aa])

        while not env.is_game_over() and step < max_step:
            env.step(action)
            next_state = env.state_id()
            reward = env.score()

            next_action_probs = epsilon_greedy_policy(
                Q, epsilon, next_state, aa, num_actions)
            expected_value = sum([next_action_probs[a] * Q[next_state][a] for a in aa])

            Q[state][action] += alpha * \
                (reward + gamma * expected_value - Q[state][action])

            state = next_state
            total_reward += reward
            step += 1

        cummulative_reward_avg.append(total_reward)
    
    return Q, cummulative_reward_avg
    
def q_learning(env,
               nb_iter: int = 10000,
               epsilon: float = 0.1,
               alpha: float = 0.5,
               gamma: float = 0.999,
               max_step: int = 1000
               ):

    num_actions = env.num_actions()
    Q = defaultdict(lambda: [0.0] * num_actions)

    cummulative_reward_avg = []

    for it in tqdm(range(nb_iter)):
        env.reset()
        state = env.state_id()
        total_reward = 0.
        step = 0

        while not env.is_game_over() and step < max_step:
            aa = env.available_actions()
            action_probs = epsilon_greedy_policy(
                Q, epsilon, state, aa, num_actions)
            action = np.random.choice(aa, p=[action_probs[a] for a in aa])

            env.step(action)
            next_state = env.state_id()
            reward = env.score()

            Q[state][action] += alpha * \
                (reward + gamma * np.max(Q[next_state][:]) - Q[state][action])

            state = next_state
            total_reward += reward
            step += 1

        cummulative_reward_avg.append(total_reward)

    return Q, cummulative_reward_avg


def dyna_q(env, nb_iter: int = 10000, alpha: float = 0.5, gamma: float = 0.999, epsilon: float = 0.1, max_step: int = 1000):

    num_actions = env.num_actions()
    num_state = env.num_states()

    Q = np.zeros((num_state, num_actions))
    model = {}

    cummulative_reward_avg = []

    for it in tqdm(range(nb_iter)):
        env = env.from_random_state()
        state = env.state_id()
        total_reward = 0.

        while not env.is_game_over():
            aa = env.available_actions()

            if np.random.rand() < epsilon:
                action = np.random.choice(aa)
            else:
                action = np.argmax(Q[state])

            env.step(action)
            reward = env.score()
            next_state = env.state_id()

            print(
                f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

            Q[state][action] += alpha * \
                (reward + gamma * np.argmax(Q[next_state]) - Q[state, action])

            model[(state, action)] = (next_state, reward)

            for _ in range(max_step):
                state, action = random.choice(list(model.keys()))
                next_state, reward = model[(state, action)]
                Q[state, action] += alpha * \
                    (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            state = next_state
            total_reward += reward

        cummulative_reward_avg.append(total_reward)

    return Q, cummulative_reward_avg









            



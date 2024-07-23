import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple
from environments import Environment


def value_iteration(env, GAMMA=0.999, THETA=0.0001):
    V = np.zeros(env.length)
    pi = np.ones((env.length, len(env.actions))) / len(env.actions)
    episode = 1

    while True:
        delta = 0.
        for s in range(env.length):
            old_v = V[s]
            best_value = float("-inf")
            for a in range(len(env.actions)):
                total = 0.
                for s_prime in range(env.length):
                    for r in range(len(env.rewards)):
                        total += env.prob_matrix[s, a, s_prime, r] * \
                            (env.rewards[r] + GAMMA * V[s_prime])
                if total > best_value:
                    best_value = total
            V[s] = best_value
            delta = max(delta, abs(V[s] - old_v))
        if delta < THETA:
            break
        episode += 1

    for s in range(env.length):
        best_value = float("-inf")
        best_action = None
        for a in range(len(env.actions)):
            total = 0.
            for s_prime in range(env.length):
                for r in range(len(env.rewards)):
                    total += env.prob_matrix[s, a, s_prime, r] * \
                        (env.rewards[r] + GAMMA * V[s_prime])
            if total > best_value:
                best_value = total
                best_action = a
        pi[s] = np.eye(len(env.actions))[best_action]

    print("Value Iteration episodes = ", episode)
    return pi, V, episode


def policy_evaluation(env: Environment, pi: np.ndarray, GAMMA: float = 0.999, THETA: float = 0.0001) -> Tuple[np.ndarray, int]:

    num_states = env.num_states()
    num_actions = env.num_actions()
    num_rewards = env.num_rewards()

    V = np.zeros(num_states)
    episode = 1

    while True:
        delta = 0.0
        for s in range(num_states):
            old_v = V[s]
            new_v = 0.0
            for a in env.available_actions():
                total_inter = 0.0
                for s_p in range(num_states):
                    for r in range(num_rewards):
                        total_inter += env.p(s, a, s_p, r) * \
                            (env.reward(r) + GAMMA * V[s_p])
                new_v += pi[s, a] * total_inter
            V[s] = new_v
            delta = max(delta, abs(V[s] - old_v))
        if delta < THETA:
            return V, episode
        episode += 1


def policy_iteration(env: Environment, GAMMA: float = 0.999, THETA: float = 0.0001) -> Tuple[np.ndarray, np.ndarray, int]:

    num_states = env.num_states()
    num_actions = env.num_actions()
    num_rewards = env.num_rewards()

    episode = 1
    pi = np.ones((num_states, num_actions)) / num_actions
    V = np.zeros(num_states)

    while True:
        # Policy Evaluation
        V, _ = policy_evaluation(env, pi, GAMMA, THETA)

        # Policy Improvement
        policy_stable = True
        for s in range(num_states):
            old_a = np.argmax(pi[s])
            action_values = np.zeros(num_actions)
            for a in env.available_actions():
                for s_p in range(num_states):
                    for r in range(num_rewards):
                        action_values[a] += env.p(s, a, s_p, r) * \
                            (env.reward(r) + GAMMA * V[s_p])
            best_a = np.argmax(action_values)
            pi[s] = np.eye(num_actions)[best_a]
            if old_a != best_a:
                policy_stable = False
        if policy_stable:
            break
        episode += 1

    print("Policy Iteration episodes = ", episode)
    return pi, V, episode


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


def monte_carlo_on_policy(
        env: Environment,
        nb_iter: int = 10000,
        GAMMA: float = 0.999,
        max_step: int = 100,
        epsilon: float = 0.1
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
        trajectory = []
        steps_count = 0
        episode_reward = 0

        while not env.is_game_over() and steps_count < max_step:
            s = env.state_id()
            aa = env.available_actions()

            if s not in pi:
                pi[s] = [
                    1.0 / len(aa) if a in aa else 0.0 for a in range(num_actions)] * num_actions

            if np.random.rand() < epsilon:
                a = np.random.choice(aa)
            else:
                a = np.argmax(pi[s])

            while env.is_forbidden(a):
                a = np.random.choice(aa)

            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score
            episode_reward += r

            trajectory.append((s, a, r, aa))
            steps_count += 1

        cummulative_reward_avg.append(episode_reward)

        G = 0
        for t, (s, a, r, aa) in reversed(list(enumerate(trajectory))):
            G = GAMMA * G + r

            if all(triplet[0] != s or triplet[1] != a for triplet in trajectory[:t]):
                Returns[(s, a)].append(G)
                Q[s][a] = np.mean(Returns[(s, a)])

                best_a = np.argmax(Q[s])
                pi[s] = [epsilon / num_actions +
                         (1 - epsilon) * (1. if i == best_a else 0.) for i in range(num_actions)]

        total_q_value = sum(sum(q) for q in Q.values())
        num_state_action_pairs = sum(len(q) for q in Q.values())
        mean_q_value = total_q_value / num_state_action_pairs
        mean_Q_value.append(mean_q_value)

    return pi, Q, cummulative_reward_avg, mean_Q_value


def create_behaviour_policy(nA):
    def policy_fn(observation, available_actions):
        A = np.zeros(nA, dtype=float)
        if len(available_actions) != 0:
            prob = 1.0 / len(available_actions)
            for action in available_actions:
                A[action] = prob

        return A
    return policy_fn


def create_target_policy(Q):
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn

def monte_carlo_off_policy(
    env,
    gamma: float = 0.999,
    nb_iter: int = 10000,
    max_step: int = 10,
    epsilon: float = 0.1
) -> Tuple[Dict[Tuple, List[float]], Dict[Tuple, List[float]]]:

    num_actions = env.num_actions()

    Q = defaultdict(lambda: np.zeros(num_actions))
    C = defaultdict(lambda: np.zeros(num_actions))

    behaviour_policy = create_behaviour_policy(num_actions)
    pi = create_target_policy(Q)

    cummulative_reward_avg = []
    mean_Q_value = []

    for _ in tqdm(range(nb_iter)):
        env.reset()
        trajectory = []
        episode_reward = 0
        step_count = 0

        s = env.state_id()

        while not env.is_game_over() and step_count < max_step:
            aa = env.available_actions()
            probs = behaviour_policy(s, aa)
            action = np.random.choice(aa, p=[probs[a] for a in aa])

            env.step(action)
            next_state = env.state_id()
            r = env.score()

            trajectory.append((s, action, r, aa))
            step_count += 1
            episode_reward += r
            s = next_state

        cummulative_reward_avg.append(episode_reward)

        G = 0
        W = 1

        # Loop for each step of the episode in reverse order
        for t in reversed(range(len(trajectory))):
            s, a, r, aa = trajectory[t]
            G = gamma * G + r

            C[s][a] += W
            Q[s][a] += (W/C[s][a]) * (G - Q[s][a])

            if a != np.argmax(pi(s)):
                break
            
            behaviour_prob = behaviour_policy(s, aa)[a]
            if behaviour_prob == 0:
                break

            W = W * (pi(s)[action]/behaviour_prob)

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
        total_reward = 0.

        action_probs = epsilon_greedy_policy(
            Q, epsilon, state, aa, num_actions)
        action = np.random.choice(aa, p=[action_probs[a] for a in aa])

        num_step = 0

        while not env.is_game_over() and num_step < max_step:
            env.step(action)
            next_state = env.state_id()
            reward = env.score()

            aa = env.available_actions()
            next_action_probs = epsilon_greedy_policy(
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

        while not env.is_game_over() and step < max_step:
            aa = env.available_actions()
            action_probs = epsilon_greedy_policy(
                Q, epsilon, state, aa, num_actions)
            action = np.random.choice(aa, p=[action_probs[a] for a in aa])

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











            



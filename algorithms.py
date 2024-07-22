import time
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple
from environments import Environment


def value_iteration(env: Environment, GAMMA: float = 0.999, THETA: float = 0.0001) -> Tuple[np.ndarray, np.ndarray, int]:

    num_states = env.num_states()
    num_actions = env.num_actions()
    num_rewards = env.num_rewards()

    V = np.zeros(num_states)
    pi = np.ones((num_states, num_actions)) / num_actions
    episode = 1

    while True:
        delta = 0.
        for s in range(num_states):
            old_v = V[s]
            best_value = float("-inf")
            for a in range(num_actions):
                total = 0.
                for s_prime in range(num_states):
                    for r in range(num_rewards):
                        total += env.p(s, a, s_prime, r) * \
                            (env.reward(r) + GAMMA * V[s_prime])
                if total > best_value:
                    best_value = total
            V[s] = best_value
            delta = max(delta, abs(V[s] - old_v))
        if delta < THETA:
            break
        episode += 1

    for s in range(num_states):
        best_value = float("-inf")
        best_action = None
        for a in range(num_actions):
            total = 0.
            for s_prime in range(num_states):
                for r in range(num_rewards):
                    total += env.p(s, a, s_prime, r) * \
                        (env.reward(r) + GAMMA * V[s_prime])
            if total > best_value:
                best_value = total
                best_action = a
        pi[s] = np.eye(num_actions)[best_action]

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
            for a in range(num_actions):
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
            for a in range(num_actions):
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

    for it in tqdm(range(nb_iter)):
        env = env.from_random_state()
        is_first_action = True
        trajectory = []
        steps_count = 0

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

    return pi, Q


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

    for it in tqdm(range(nb_iter)):
        env = env.from_random_state()
        trajectory = []
        steps_count = 0

        while not env.is_game_over() and steps_count < max_step:
            s = env.state_id()
            aa = env.available_actions()

            if s not in pi:
                pi[s] = [1.0 / num_actions] * num_actions

            if np.random.rand() < epsilon:
                a = np.random.choice(aa)
            else:
                a = np.argmax(pi[s])

            while env.is_forbidden(a):
                aa = [a for a in aa if not env.is_forbidden(a)]
                a = np.random.choice(aa)

            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score

            trajectory.append((s, a, r, aa))
            steps_count += 1

        G = 0
        for t, (s, a, r, aa) in reversed(list(enumerate(trajectory))):
            G = GAMMA * G + r

            if all(triplet[0] != s or triplet[1] != a for triplet in trajectory[:t]):
                Returns[(s, a)].append(G)
                Q[s][a] = np.mean(Returns[(s, a)])

                best_a = np.argmax(Q[s])
                pi[s] = [epsilon / num_actions +
                         (1 - epsilon) * (1. if i == best_a else 0.) for i in range(num_actions)]

    return pi, Q


def monte_carlo_off_policy(
    env,
    gamma: float = 0.999,
    nb_iter: int = 10000,
    max_steps: int = 10,
    epsilon: float = 0.1
) -> Tuple[Dict[Tuple, List[float]], Dict[Tuple, List[float]]]:
    Q: Dict[Tuple, List[float]] = defaultdict(
        lambda: [0.0] * len(env.available_actions()))
    pi: Dict[Tuple, List[float]] = defaultdict(
        lambda: [1.0 / len(env.available_actions())] * len(env.available_actions()))
    C: Dict[Tuple, float] = defaultdict(float)

    for s in Q:
        best_action = np.argmax(Q[s])
        pi[s] = [1.0 if i == best_action else 0.0 for i in range(len(Q[s]))]

    for _ in tqdm(range(nb_iter)):
        env = env.from_random_state()
        trajectory = []
        steps_count = 0

        while not env.is_done() and steps_count < max_steps:
            s = env.state_id()
            aa = env.available_actions()
            n_actions = len(aa)

            if np.random.rand() < epsilon:
                a = np.random.choice(aa)
            else:
                a = np.argmax(pi[s])

            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score

            trajectory.append((s, a, r, aa))
            steps_count += 1

        G = 0
        W = 1

        # Loop for each step of the episode in reverse order
        for t in reversed(range(len(trajectory))):
            s, a, r, aa = trajectory[t]
            G = gamma * G + r

            C[(s, a)] += W
            Q[s][a] += (W / C[(s, a)]) * (G - Q[s][a])

            best_action = np.argmax(Q[s])
            pi[s] = [1.0 if i == best_action else 0.0 for i in range(len(aa))]

            if a != best_action:
                break

            W *= 1 / (epsilon / n_actions + (1 - epsilon)
                      * (1 if a == best_action else 0))

    return pi, Q









            



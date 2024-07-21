import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple
from environments import Environment


def value_iteration(env: Environment, GAMMA: float = 0.999, THETA: float = 0.0001) -> Tuple[np.ndarray, np.ndarray, int]:
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
    V = np.zeros(env.length)
    episode = 1

    while True:
        delta = 0.0
        for s in range(env.length):
            old_v = V[s]
            new_v = 0.0
            for a in range(len(env.actions)):
                total_inter = 0.0
                for s_p in range(env.length):
                    for r in range(len(env.rewards)):
                        total_inter += env.prob_matrix[s, a, s_p,
                                                       r] * (env.rewards[r] + GAMMA * V[s_p])
                new_v += pi[s, a] * total_inter
            V[s] = new_v
            delta = max(delta, abs(V[s] - old_v))
        if delta < THETA:
            return V, episode
        episode += 1


def policy_iteration(env: Environment, GAMMA: float = 0.999, THETA: float = 0.0001) -> Tuple[np.ndarray, np.ndarray, int]:
    episode = 1
    pi = np.ones((env.length, len(env.actions))) / len(env.actions)
    V = np.zeros(env.length)

    while True:
        # Policy Evaluation
        V, _ = policy_evaluation(env, pi, GAMMA, THETA)

        # Policy Improvement
        policy_stable = True
        for s in range(env.length):
            old_a = np.argmax(pi[s])
            action_values = np.zeros(len(env.actions))
            for a in range(len(env.actions)):
                for s_p in range(env.length):
                    for r in range(len(env.rewards)):
                        action_values[a] += env.prob_matrix[s, a,
                                                            s_p, r] * (env.rewards[r] + GAMMA * V[s_p])
            best_a = np.argmax(action_values)
            pi[s] = np.eye(len(env.actions))[best_a]
            if old_a != best_a:
                policy_stable = False
        if policy_stable:
            break
        episode += 1

    print("Policy Iteration episodes = ", episode)
    return pi, V, episode


def monte_carlo_with_exploring_start(env: Environment, nb_iter: int = 10000, GAMMA: float = 0.999, max_step: int = 10) -> Tuple[Dict[Tuple, List[float]], Dict[Tuple, List[float]]]:
    Q: Dict[Tuple, List[float]] = defaultdict(
        lambda: [0.0] * len(env.available_actions()))
    pi: Dict[Tuple, List[float]] = defaultdict(
        lambda: [1.0 / len(env.available_actions())] * len(env.available_actions()))
    Returns: Dict[Tuple, List[float]] = defaultdict(list)

    for it in tqdm(range(nb_iter)):
        env = env.from_random_state()
        is_first_action = True
        trajectory = []
        steps_count = 0

        while not env.is_done() and steps_count < max_step:
            s = env.state_id()
            aa = env.available_actions()

            if s not in pi:
                pi[s] = [1.0 / len(aa)] * len(aa)

            if is_first_action:
                a = np.random.choice(aa)
                is_first_action = False
            else:
                a = np.argmax(pi[s])

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
                for a in aa:
                    if (s, a) not in Q:
                        Q[s][a] = np.random.random()
                    if Q[s][a] > best_a_score:
                        best_a = a
                        best_a_score = Q[s][a]

                pi[s] = [1.0 if i == best_a else 0.0 for i in range(len(aa))]

    return pi, Q


def monte_carlo_on_policy(
        env: Environment,
        nb_iter: int = 10000,
        GAMMA: float = 0.999,
        max_step: int = 100,
        epsilon: float = 0.1
) -> Tuple[Dict[Tuple, List[float]], Dict[Tuple, List[float]]]:

    Q: Dict[Tuple, List[float]] = defaultdict(
        lambda: [0.0] * len(env.available_actions()))
    pi: Dict[Tuple, List[float]] = defaultdict(
        lambda: [1.0 / len(env.available_actions())] * len(env.available_actions()))
    Returns: Dict[Tuple, List[float]] = defaultdict(list)

    for it in tqdm(range(nb_iter)):
        env = env.from_random_state()
        trajectory = []
        steps_count = 0

        while not env.is_done() and steps_count < max_step:
            s = env.state_id()
            aa = env.available_actions()
            nb_actions = len(aa)

            if s not in pi:
                pi[s] = [1.0 / nb_actions] * nb_actions

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
        for t, (s, a, r, aa) in reversed(list(enumerate(trajectory))):
            G = GAMMA * G + r

            if all(triplet[0] != s or triplet[1] != a for triplet in trajectory[:t]):
                Returns[(s, a)].append(G)
                Q[s][a] = np.mean(Returns[(s, a)])

                nb_actions = len(aa)
                best_a = np.argmax(Q[s])
                pi[s] = [epsilon / nb_actions +
                         (1 - epsilon) * (1. if i == best_a else 0.) for i in range(nb_actions)]

    return pi, Q











            



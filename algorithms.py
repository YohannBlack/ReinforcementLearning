import numpy as np
from tqdm import tqdm
from collections import defaultdict


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


def policy_evaluation(env, pi, GAMMA=0.999, THETA=0.0001):
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


def policy_iteration(env, GAMMA=0.999, THETA=0.0001):
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


def naive_monte_carlo_with_exploring_starts(env, gamma=0.999, nb_iter=10000, max_steps=10):
    Pi = {}
    Q = {}
    Returns = {}

    for it in tqdm(range(nb_iter)):
        env = env.from_random_state()

        is_first_action = True
        trajectory = []
        steps_count = 0
        while not env.is_done() and steps_count < max_steps:
            s = env.state_id()
            A = env.available_actions()

            if s not in Pi:
                Pi[s] = np.random.choice(A)

            if is_first_action:
                a = np.random.choice(A)
                is_first_action = False
            else:
                a = Pi[s]

            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score

            trajectory.append((s, a, r, A))
            steps_count += 1

        G = 0
        for (t, (s, a, r, A)) in reversed(list(enumerate(trajectory))):
            G = gamma * G + r

            if all(map(lambda triplet: triplet[0] != s or triplet[1] != a, trajectory[:t])):
                if (s, a) not in Returns:
                    Returns[(s, a)] = []
                Returns[(s, a)].append(G)
                Q[(s, a)] = np.mean(Returns[(s, a)])

                best_a = None
                best_a_score = -np.inf
                for a in A:
                    if (s, a) not in Q:
                        Q[(s, a)] = np.random.random()
                    if Q[(s, a)] > best_a_score:
                        best_a = a
                        best_a_score = Q[(s, a)]

                Pi[s] = best_a
    return Pi


def monte_carlo_on_policy(env, gamma=0.999, nb_iter=10000, epsilon=0.1, max_steps=10):
    Q = defaultdict(lambda: np.zeros(len(env.available_actions())))
    Returns = defaultdict(list)

    def epsilon_soft_policy(s):
        A_star = np.argmax(Q[s])
        policy = np.ones(len(env.available_actions())) * \
            (epsilon / len(env.available_actions()))
        policy[A_star] += (1 - epsilon)
        return policy

    def choose_action(policy):
        return np.random.choice(np.arange(len(policy)), p=policy)

    def generate_episode(policy):
        episode = []
        state = env.from_random_state().state_id()
        steps_count = 0
        while not env.is_done() and steps_count < max_steps:
            action = choose_action(policy[state])
            s_prime, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = s_prime
            steps_count += 1
        return episode

    for _ in tqdm(range(nb_iter)):
        episode = generate_episode(epsilon_soft_policy)








            



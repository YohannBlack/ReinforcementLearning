import numpy as np
from tqdm import tqdm


def value_iteration(env, gamma=0.9, theta=0.0001):
    V = np.zeros(env.length)
    policy = np.zeros((env.length, len(env.actions)))

    while True:
        delta = 0
        for s in range(env.length):
            v = V[s]
            new_values = np.zeros(len(env.actions))
            for a in env.actions:
                for s_prime in range(env.length):
                    prob = env.prob_matrix[s, a, s_prime]
                    reward = env.goal_reward if s_prime == env.goal_position else env.step_reward
                    new_values[a] += prob * (reward + gamma * V[s_prime])
            V[s] = np.max(new_values)
            policy[s] = np.eye(len(env.actions))[np.argmax(new_values)]
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break
    return policy, V


def policy_iteration(env, gamma=0.9, theta=0.0001):
    V = np.zeros(env.length)
    policy = np.full((env.length, len(env.actions)), 1/len(env.actions))

    def one_step_lookahead(s, V):
        action_values = np.zeros(len(env.actions))
        for a in range(len(env.actions)):
            for s_prime in range(env.length):
                for r in range(len(env.rewards)):
                    prob = env.prob_matrix[s, a, s_prime, r]
                    reward = env.rewards[r]
                    action_values[a] += prob * (reward + gamma * V[s_prime])
        return action_values

    is_policy_stable = False

    while not is_policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(env.length):
                v = V[s]
                V[s] = sum(policy[s, a] * sum(env.prob_matrix[s, a, s_prime, r] * (env.rewards[r] + gamma * V[s_prime])
                    for s_prime in range(env.length)
                    for r in range(len(env.rewards))
                ) for a in range(len(env.actions)))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        is_policy_stable = True
        for s in range(env.length):
            old_action = np.argmax(policy[s])
            action_values = one_step_lookahead(s, V)
            best_action = np.argmax(action_values)

            new_policy = np.zeros(len(env.actions))
            new_policy[best_action] = 1.0
            policy[s] = new_policy

            if old_action != best_action:
                is_policy_stable = False

    return policy, V


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






            



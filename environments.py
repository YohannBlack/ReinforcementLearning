import pygame
import numpy as np 
from abc import ABC, abstractmethod


class Environment(ABC):
    @abstractmethod
    def num_states(self) -> int:
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def num_actions(self) -> int:
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def num_rewards(self) -> int:
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def reward(self, i: int) -> float:
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def state_id(self) -> int:
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def display(self):
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def is_forbidden(self, action: int) -> int:
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def is_game_over(self) -> bool:
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def available_actions(self) -> np.ndarray:
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def step(self, action: int):
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def score(self):
        raise NotImplementedError("This method is not implemented")

    @staticmethod
    @abstractmethod
    def from_random_state() -> 'Environment':
        raise NotImplementedError("This method is not implemented")


class LineWorld(Environment):
    def __init__(self, length=10):
        super().__init__()
        self.length = length
        self.state = 0
        self.actions = [0, 1]
        self.rewards = [-1, 0]
        self.goal_position = self.length - 1
        self.goal_color = (0, 255, 0)
        self.agent_color = (255, 0, 0)
        self.start_color = (0, 0, 255)
        self.prob_matrix = self.initiate_prob_matrix()

    def from_random_state(self) -> 'LineWorld':
        env = LineWorld(length=self.length)
        env.state = np.random.randint(0, env.length)
        return env
    
    def initiate_prob_matrix(self):
        p = np.zeros((self.length, len(self.actions), self.length, len(self.rewards)))

        for s in range(self.length):
            for a in range(len(self.actions)):
                s_prime = s + self.actions[a]
                if 0 <= s_prime < self.length:
                    for r in range(len(self.rewards)):
                        if s_prime == self.goal_position and self.rewards[r] == 0:
                            p[s, a, s_prime, r] = 1.0
                        elif s_prime != self.goal_position and self.rewards[r] == -1:
                            p[s, a, s_prime, r] = 1.0
        
        return p
                    
    def num_states(self) -> int:
        return self.length

    def num_actions(self) -> int:
        return len(self.actions)

    def num_rewards(self) -> int:
        return len(self.rewards)

    def reward(self, i: int) -> float:
        return self.rewards[i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return self.prob_matrix[s, a, s_p, r_index]

    def state_id(self):
        return self.state

    def reset(self):
        self.state = 0
        return self.state
    
    def display(self):
        pass

    def is_forbidden(self, action: int) -> int:
        if action == 0 and self.state == self.length - 1:
            return 1
        elif action == 1 and self.state == 0:
            return 1
        return 0

    def is_game_over(self) -> bool:
        return self.state == self.length - 1

    def available_actions(self):
        return [a for a in self.actions if not self.is_forbidden(a)]

    def step(self, action):
        old_state = self.state

        if action == 1:
            self.state = min(old_state + 1, self.goal_position)
        elif action == 0:
            self.state = max(old_state - 1, 0)

        if self.state == self.goal_position:
            reward = self.rewards[1]
            done = True
        else:
            reward = self.rewards[0]
            done = False

    def score(self):
        if self.state == self.goal_position:
            return 1

        return -1
    
    def render(self, screen):
        screen.fill((255, 255, 255))
        for i in range(self.length):
            pygame.draw.rect(screen, (0, 0, 0), (i * 50, 0, 50, 50), 1)
            if i == self.state:
                pygame.draw.rect(screen, self.agent_color, (i * 50, 0, 50, 50))
            if i == self.start_color:
                pygame.draw.rect(screen, self.start_color, (i * 50, 0, 50, 50))
            if i == self.goal_position:
                pygame.draw.rect(screen, self.goal_color, (i * 50, 0, 50, 50))
        pygame.display.flip()


class GridWorld(Environment):
    def __init__(self, width: int, height: int) -> Environment:
        super().__init__()
        self.width = width
        self.height = height
        self.length = width * height
        self.state = 0
        self.actions = [0, 1, 2, 3]
        self.rewards = [-1., 1.]
        self.goal_position = (self.width - 1, self.height - 1)
        self.goal_color = (0, 255, 0)
        self.agent_color = (255, 0, 0)
        self.start_color = (0, 0, 255)
        self.prob_matrix = self.initiate_prob_matrix()

    def from_random_state(self) -> 'GridWorld':
        env = GridWorld(width=self.width, height=self.height)
        env.state = np.random.randint(0, env.length)
        return env

    def initiate_prob_matrix(self):
        p = np.zeros((self.length, len(self.actions),
                     self.length, len(self.rewards)))

        for s in range(self.length):
            for a in range(len(self.actions)):
                s_prime = self.next_state(s, a)
                if 0 <= s_prime < self.length and s_prime != s:
                    for r in range(len(self.rewards)):
                        if s_prime == self.goal_position and self.rewards[r] == 1.:
                            p[s, a, s_prime, r] = 1.0
                        elif s_prime != self.goal_position and self.rewards[r] == -1.:
                            p[s, a, s_prime, r] = 1.0
        return p

    def next_state(self, s, action):
        row, col = divmod(s, self.width)
        if action == 0 and row > 0:  # Up
            row -= 1
        elif action == 1 and row < self.height - 1:  # Down
            row += 1
        elif action == 2 and col > 0:  # Left
            col -= 1
        elif action == 3 and col < self.width - 1:  # Right
            col += 1
        return row * self.width + col

    def num_states(self) -> int:
        return self.length

    def num_actions(self) -> int:
        return len(self.actions)

    def num_rewards(self) -> int:
        return len(self.rewards)

    def reward(self, i: int) -> float:
        return self.rewards[i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return self.prob_matrix[s, a, s_p, r_index]

    def state_id(self):
        return self.state

    def reset(self):
        self.state = 0
        return self.state

    def display(self):
        pass

    def is_forbidden(self, action: int) -> int:
        row, col = divmod(self.state, self.width)
        if (action == 0 and row == 0) or (action == 1 and row == self.height - 1) or \
           (action == 2 and col == 0) or (action == 3 and col == self.width - 1):
            return 1
        return 0

    def is_game_over(self) -> bool:
        return self.state == self.goal_position

    def available_actions(self) -> np.ndarray:
        return [a for a in self.actions if not self.is_forbidden(a)]

    def step(self, action):
        if self.is_forbidden(action):
            raise ValueError("Invalid action")

        old_state = self.state
        self.state = self.next_state(old_state, action)

    def score(self):
        if self.state == self.goal_position:
            return self.rewards[1]
        return self.rewards[0]

    def render(self, screen):
        screen.fill((255, 255, 255))
        cell_size = min(screen.get_width() // self.width,
                        screen.get_height() // self.height)
        for row in range(self.height):
            for col in range(self.width):
                i = row * self.width + col
                pygame.draw.rect(screen, (0, 0, 0),
                                 (col * cell_size, row * cell_size, cell_size, cell_size), 1)
                if i == self.start_color:
                    pygame.draw.rect(screen, self.start_color,
                                     (col * cell_size, row * cell_size, cell_size, cell_size))
                if i == self.state:
                    pygame.draw.rect(screen, self.agent_color,
                                     (col * cell_size, row * cell_size, cell_size, cell_size))
                if i == self.goal_position:
                    pygame.draw.rect(screen, self.goal_color,
                                     (col * cell_size, row * cell_size, cell_size, cell_size))
        pygame.display.flip()


class TwoRoundRockPaperScissors(Environment):
    def __init__(self):
        super().__init__()
        self.state = 0
        self.actions = [0, 1, 2]  # Rock, Paper, Scissors
        self.rewards = [-1, 0, 1]
        self.agent_first_round = None
        self.opponent_first_round_action = None
        self.opponent_second_round_action = None

    def from_random_state(self) -> 'TwoRoundRockPaperScissors':
        return TwoRoundRockPaperScissors()

    def num_states(self) -> int:
        return 2

    def num_actions(self) -> int:
        return len(self.actions)

    def num_rewards(self) -> int:
        return len(self.rewards)

    def reward(self, i: int) -> float:
        return self.rewards[i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s == 0 and s_p == 1:
            return 1.0 if a in self.actions else 0.0
        elif s == 1 and s_p == 2:
            return 1.0 if a in self.actions else 0.0
        return 0.0

    def state_id(self) -> int:
        return self.state

    def reset(self):
        self.state = 0
        self.agent_first_round = None
        self.opponent_first_round_action = None
        self.opponent_second_round_action = None
        return self.state

    def display(self):
        print(f"Current state: {self.state}")
        print(f"Agent's first round action: {self.agent_first_round_action}")
        print(
            f"Opponent's first round action: {self.opponent_first_round_action}")
        print(
            f"Opponent's second round action: {self.opponent_second_round_action}")

    def is_forbidden(self, action: int) -> int:
        return 0

    def is_game_over(self) -> bool:
        return self.state == 2

    def available_actions(self) -> np.ndarray:
        if self.state == 0:
            return self.actions
        if self.state == 1:
            return self.actions
        return []

    def step(self, action: int):
        if self.state == 0:
            self.agent_first_round = action
            self.opponent_first_round_action = np.random.choice(self.actions)
            self.state = 1
        elif self.state == 1:
            self.opponent_second_round_action = self.agent_first_round
            self.state = 2
        else:
            raise ValueError("Game is over")

    def score(self):
        return

    def _calculate_score(self, agent_action, opponent_action):
        if agent_action == opponent_action:
            return self.rewards[1]
        elif (agent_action == 0 and opponent_action == 2) or \
             (agent_action == 1 and opponent_action == 0) or \
             (agent_action == 2 and opponent_action == 1):
            return self.rewards[2]
        return self.rewards[0]

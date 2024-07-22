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
        return self.actions

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
            if i == self.goal_position:
                pygame.draw.rect(screen, self.goal_color, (i * 50, 0, 50, 50))
        pygame.display.flip()


class GridWorld(Environment):
    def __init__(self, width=10, height=10):
        super().__init__()
        self.width = width
        self.height = height
        self.length = width * height
        self.state = 0
        self.actions = [0, 1, 2, 3]
        self.rewards = [-1.0, 1.0]
        self.goal_position = self.length - 1
        self.goal_color = (0, 255, 0)
        self.agent_color = (255, 0, 0)
        self.prob_matrix = self.initiate_prob_matrix()

    def from_random_state(self):
        env = GridWorld(width=self.width, height=self.height)
        env.state = np.random.randint(0, self.length)
        return env

    def initiate_prob_matrix(self):
        p = np.zeros((self.length, len(self.actions),
                     self.length, len(self.rewards)))

        for s in range(self.length):
            x, y = self._state_to_coordinate(s)
            for a in range(len(self.actions)):

                if a == 0:
                    new_x, new_y = x, y + 1
                elif a == 1:
                    new_x, new_y = x, y - 1
                elif a == 2:
                    new_x, new_y = x + 1, y
                elif a == 3:
                    new_x, new_y = x - 1, y

                if self._is_valid_state(new_x, new_y):
                    s_prime = self._coordinate_to_state(new_x, new_y)
                else:
                    s_prime = s

                if s_prime == self.goal_position:
                    p[s, a, s_prime, 1] = 1.0
                else:
                    p[s, a, s_prime, 0] = 1.0
        return p

    def _is_valid_state(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def _state_to_coordinate(self, state):
        # x, y = divmod(state, self.width)
        x, y = state % self.width, state // self.width
        return x, y

    def _coordinate_to_state(self, x, y):
        return y * self.width + x

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        old_state = self.state
        x, y = self._state_to_coordinate(old_state)

        if action == 0:
            new_x, new_y = x, y + 1
        elif action == 1:
            new_x, new_y = x, y - 1
        elif action == 2:
            new_x, new_y = x + 1, y
        elif action == 3:
            new_x, new_y = x - 1, y

        if self._is_valid_state(new_x, new_y):
            self.state = self._coordinate_to_state(new_x, new_y)
        else:
            self.state = old_state

        if self.state == self.goal_position:
            reward = self.rewards[1]
            done = True
        else:
            reward = self.rewards[0]
            done = False

        return self.state, reward, done

    def render(self, screen):
        screen.fill((255, 255, 255))
        cell_size = min(screen.get_width() // self.width,
                        screen.get_height() // self.height)

        for y in range(self.height):
            for x in range(self.width):
                pygame.draw.rect(screen,
                                 (0, 0, 0),
                                 (x * cell_size, y * cell_size,
                                  cell_size, cell_size),
                                 1)
                current_state = self._coordinate_to_state(x, y)
                if current_state == self.state:
                    pygame.draw.rect(screen,
                                     self.agent_color,
                                     (x * cell_size, y * cell_size, cell_size, cell_size))
                if current_state == self.goal_position:
                    pygame.draw.rect(screen,
                                     self.goal_color,
                                     (x * cell_size, y * cell_size, cell_size, cell_size))

        pygame.display.flip()

    def is_done(self):
        return self.state == self.goal_position

    def score(self):
        if self.state == self.goal_position:
            return 1

        return -1

    def state_id(self):
        return self.state

    def available_actions(self):
        return self.actions





        


"""
Implementation of the following paper:

Sutton, R.S. (1981). “Adaptation of learning rate parameters.”
In: Goal Seeking Components for Adaptive Intelligence: An Initial Assessment, by A. G. Barto and R. S. Sutton.
Air Force Wright Aeronautical Laboratories Technical Report AFWAL-TR-81-1070.
Wright-Patterson Air Force Base, Ohio 45433.
"""

import numpy as np


class Agent:
    def __init__(self, initial_gain: float, b: float, gain_depth: int):
        """
        :param initial_gain:
        :param b: learning rate control parameter in the last layer.
        :param gain_depth: number of levels for gain.
        """
        self.gain = initial_gain
        self.y_t = 0
        self.b = b

        self.pe_t = 0
        self.pe_t_minus_1 = 0
        self.pe_product = 0  # place holder for pe_t * pe_t_minus_1

        # initialize a list of gains where the first value in the list corresponds to
        # the gain of level 0 (i.e., top level learning rate/gain), and the second last value corresponds to
        # the gain of level 1 (i.e., gain of the level 0 dynamic gain), etc.
        self.gain_depth = gain_depth
        self.gain_hierarchy_values: list = [0] * (self.gain_depth + 1)

    def action(self, true_y_t: float) -> float:
        """
        Agent's action.

        :param true_y_t: provided by the environment
        :return:
        """
        self.pe_t_minus_1 = self.pe_t  # E(t-1) equation C.2
        self.pe_t = true_y_t - self.y_t  # E(t) equation C.2

        if self.gain_depth != 0:
            self.compute_gain()

        self.y_t += self.gain * self.pe_t

        return self.y_t

    def compute_gain(self):
        """
        iteratively update each level gain in the gain value list.
        The deepest gain, which is controlled directly by the
        control parameter corresponds to the last value of the list.
        Top level gain is the first value in the list.
        :return:
        """
        self.pe_product = self.pe_t * self.pe_t_minus_1
        self.gain_hierarchy_values[-1] = self.b

        for idx in range(self.gain_depth - 1, -1, -1):  # loop from the second last index to index 0.
            self.gain_hierarchy_values[idx] += self.gain_hierarchy_values[idx + 1] * self.pe_product  # equation C.2

        self.gain = self.gain_hierarchy_values[0]


class SimulatedEnvironment:
    def __init__(self, sa: float, sb: float, max_steps: int = 100):
        """
        Simulated random walk environment.
        Section C.4 in the paper.

        :param sa: variance for A(t) in random walk z(t).
        :param sb: variance for B(t) in random movement y(t).
        """
        self.z_t = 0
        self.y_t = 0
        self.sa = sa
        self.sb = sb
        self.a_t = 0
        self.b_t = 0

        self.max_steps = max_steps

        # Pre-compute the random normal values for efficiency
        self.random_normal_list_1 = np.random.normal(loc=0.0, scale=self.sa, size=max_steps)
        self.random_normal_list_2 = np.random.normal(loc=0.0, scale=self.sb, size=max_steps)

        self.internal_count = 0

    def step(self) -> float:
        """
        Environment that is independent of agents' actions.
        Equation C.3 and C.4 in the paper.
        :return: y_t
        """
        self.a_t = self.random_normal_list_1[self.internal_count]
        self.b_t = self.random_normal_list_2[self.internal_count]

        self.y_t = self.z_t + self.b_t  # equation C.3
        self.z_t += self.a_t  # equation C.4

        self.internal_count += 1

        return float(self.y_t)

    def change_variance(self, sb):
        self.random_normal_list_2 = np.random.normal(loc=0.0, scale=sb, size=self.max_steps)


def main():
    gain_list = []
    max_steps = 3000

    env = SimulatedEnvironment(sa=1.0, sb=3.0, max_steps=max_steps)
    agent = Agent(initial_gain=0.0, b=0.001, gain_depth=1)

    for step in range(max_steps):
        true_y_t = env.step()
        _ = agent.action(true_y_t=true_y_t)
        gain_list.append(agent.gain)


if __name__ == '__main__':
    main()
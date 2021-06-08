import numpy as np
import gym
from gym import spaces
import random
from gym.utils import seeding


class conn(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    # Define constants for clearer code

    def __init__(self, connectors_dicc, list_signals, connector, len_test_conn, max_categories, state_space=10000):
        """
        Init method
        :param connectors_dicc:
        :param list_signals:
        :param connector:
        :param len_test_conn:
        :param max_categories:
        :param state_space:
        """
        super(conn, self).__init__()
        # numero de pins connector -1 due to PartNumber
        self.grid_size = len_test_conn
        self.connectors_dicc = connectors_dicc
        self.signals = list_signals  # different signals to plug into a PIN Categories
        # Example when using discrete actions,
        self.state_space = state_space
        self.max_categories = max_categories
        # space number of pins of this connector
        self.action_space = spaces.Discrete(self.state_space)
        # observation space low/High bound number of diferent signals this connector
        # size, vector length number of pins
        self.observation_space = spaces.Box(low=0, high=self.max_categories,
                                            shape=(self.grid_size,), dtype=np.float32)
        # estate list of actions
        self.state = list(np.zeros(self.grid_size, dtype='int'))
        self.agent_pos = 0
        self.terminal = False
        self.wirecolors = []
        self.signal_groups = []
        self.lr = .5
        self.this_connector = connector
        self.reward = 0
        self.already_pinned = []
        self.already_pinned_thickness = []
        self._seed()
        self.signals_cats = []
        self.already_pinned_in_out = []
        self.multicore_pinned = []
        self.signals_category_dicc = dicci = eval(self.connectors_dicc[self.this_connector]['1']['dicc_signals'])
        for key in self.connectors_dicc[self.this_connector].keys():
            # list with real categories of this conector
            if key != "PartNumber":
                self.signals_cats.append(self.connectors_dicc[self.this_connector][key]['Cat_Signal'])
        return

    def _seed(self, seed=None):
        """
        define seed
        :param seed:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        :return:
        """
        self.state = np.zeros(self.grid_size, dtype='int')

        self.agent_pos = 0
        self.reward = 0
        self.wirecolors = []
        self.signal_groups = []
        self.already_pinned = []
        self.already_pinned_thickness = []
        self.already_pinned_in_out = []
        self.multicore_pinned = []

        return self.state

    def render(self, mode='console'):
        """
        render method no implemented for now
        :param mode:
        :return:
        """
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot

        print("x", end="")

    def get_reward_cat(self, reward, agent_pos, signal_group, neighbors):
        """
        Get regards categories
        :param reward:
        :param agent_pos:
        :param signal_group:
        :param neighbors:
        :return:
        """
        # save the previous rewards
        old_reward = reward
        for pin in neighbors:
            if pin in self.already_pinned:
                # get signal group of this neigbourg
                # 'BATT_'1, 'CAN_' 2, 'CANFD_' 3, 'GND_' 4, 'GNDR_GND_' 5,'HV_' 6, else 7
                signal_group_prev = self.connectors_dicc[self.this_connector][str(pin)]['signal_group']

                # Don't put power and ground next to each other   # 1--> battery, 6--> High Power 4,5 Ground
                if (signal_group_prev == 1 and signal_group == 5) or (signal_group_prev == 1 and signal_group == 4):
                    reward = reward - 4
                elif (signal_group_prev == 5 and signal_group == 1) or (signal_group_prev == 4 and signal_group == 1):
                    reward = reward - 4
                if (signal_group_prev == 6 and signal_group == 5) or (signal_group_prev == 6 and signal_group == 4):
                    reward = reward - 8
                elif (signal_group_prev == 5 and signal_group == 6) or (signal_group_prev == 4 and signal_group == 6):
                    reward = reward - 8

                # Don't put power and information next to each other   # 1--> battery, 6--> High Power 2,3 Ground
                if (signal_group_prev == 1 and signal_group == 3) or (signal_group_prev == 1 and signal_group == 2):
                    reward = reward - 4
                elif (signal_group_prev == 3 and signal_group == 1) or (signal_group_prev == 2 and signal_group == 1):
                    reward = reward - 4
                if (signal_group_prev == 6 and signal_group == 3) or (signal_group_prev == 6 and signal_group == 2):
                    reward = reward - 8
                elif (signal_group_prev == 3 and signal_group == 6) or (signal_group_prev == 2 and signal_group == 6):
                    reward = reward - 8

        if old_reward == reward:
            reward = reward + 8
        return reward

    def get_reward_colorwire(self, reward, agent_pos, colorwire, neighbors, thickness):
        """
        get wire Color Reward
        :param reward:
        :param agent_pos:
        :param colorwire:
        :param neighbors:
        :param thickness:
        :return:
        """
        # save the previous rewards
        old_reward = reward
        for pin in neighbors:
            if pin in self.already_pinned:
                if colorwire != "na" and colorwire in self.wirecolors and thickness in self.already_pinned_thickness:
                    reward = reward - 4
                elif colorwire != "na" and colorwire in self.wirecolors and not (
                        thickness in self.already_pinned_thickness):
                    reward = reward + 2
                elif colorwire != "na":
                    reward = reward + 1

        if old_reward == reward:
            reward = reward + 8
        return reward

    def get_reward_multicore(self, reward, agent_pos, multicore_label, neighbors, in_ext):
        """
        get reward multicore wires
        :param reward:
        :param agent_pos:
        :param multicore_label:
        :param neighbors:
        :param in_ext:
        :return:
        """
        # save the previous rewards
        old_reward = reward
        for pin in neighbors:
            if pin in self.already_pinned:
                # multicore with pair in neigbhors and pining to outside
                if multicore_label != "na" and multicore_label in self.multicore_pinned and in_ext == "False":
                    reward = reward + 4
                # multicore with pair in neigbhors and NOT pining to outside
                elif multicore_label != "na" and multicore_label in self.multicore_pinned and in_ext == "True":
                    reward = reward - 4
                # multicore with NOT pair in neigbhors and pining to outside
                elif multicore_label != "na" and not (multicore_label in self.multicore_pinned) and in_ext == "False":
                    reward = reward - 2
                elif multicore_label != "na" and not (multicore_label in self.multicore_pinned) and in_ext == "True":
                    reward = reward - 4
                elif multicore_label != "na":
                    reward = reward + 1
        # if not multicore on the neighbors lets assume that is a new multicore
        if old_reward == reward and multicore_label != "na":
            reward = reward + 4
        return reward

    def get_reward_internal_external(self, reward, agent_pos, in_ext, thickness):
        """
        get reward pin interior / exterio cavity
        :param reward:
        :param agent_pos:
        :param in_ext:
        :param thickness:
        :return:
        """
        # Thick wires outside and thin wires internal if exists internal cavities
        # we conside a thick wire > 1

        if in_ext == "False" and thickness <= 1:
            reward = reward - 2

        elif in_ext == "True" and thickness <= 1:
            reward = reward + 2
        elif in_ext == "False" and thickness > 1:
            reward = reward + 2
        elif in_ext == "True" and thickness > 1:
            reward = reward - 1
        return reward

    def calculate_reward(self, combination):
        """
        Main Calculate reward function
        :param combination:
        :return:
        """
        reward = 0

        signal = combination[self.agent_pos]  # signal category (action = pin this wire with this signal)
        # check if this signal was pinned in this Pin

        if self.connectors_dicc[self.this_connector][str(self.agent_pos + 1)]['Cat_Signal'] == signal:
            reward = reward + 10
        else:
            reward = reward - 10

        # WireColor
        colorwire = self.connectors_dicc[self.this_connector][str(self.agent_pos + 1)]['WireColor']
        # thickness
        thickness = self.connectors_dicc[self.this_connector][str(self.agent_pos + 1)]['WireCSA']
        # signal Group
        signal_group = self.connectors_dicc[self.this_connector][str(self.agent_pos + 1)]['signal_group']
        # neighbors
        neighbors = self.connectors_dicc[self.this_connector][str(self.agent_pos + 1)]['neighbors']
        # internal/external PIN
        in_ext = self.connectors_dicc[self.this_connector][str(self.agent_pos + 1)]['internal_pin']
        # Mulicore Features
        multicore_label = self.connectors_dicc[self.this_connector][str(self.agent_pos + 1)]['Multicore']

        if self.agent_pos + 1 > 1:
            # get position signal reward
            reward = self.get_reward_cat(reward, self.agent_pos + 1, signal_group, neighbors)
            # get colorwire reward
            reward = self.get_reward_colorwire(reward, self.agent_pos + 1, colorwire, neighbors, thickness)
            # get internal/external rewars
            reward = self.get_reward_internal_external(reward, self.agent_pos + 1, in_ext, thickness)
            # get Mukticore reward
            reward = self.get_reward_multicore(reward, self.agent_pos + 1, multicore_label, neighbors, in_ext)

        return reward

    def get_real(self, action):
        """
        get real position for this action if it was ever seen in this connector
        :param action:
        :return:
        """
        real = 0
        if str(action) in self.signals_cats:
            real = self.signals_cats.index(action) + 1
        else:
            real = self.agent_pos + 1
        return real

    def step(self, action):
        """
        step method. take an action and evaluate the reward
        :param action:
        :return:
        """

        # Update Action
        self.state[self.agent_pos] = action
        real = self.get_real(action)  # real values
        # calculate Reward
        self.reward = self.reward + self.calculate_reward(self.state)
        print(self.reward)

        self.already_pinned.append(self.agent_pos + 1)  # list control which pin are already in use
        # self.terminal = bool(self.agent_pos >= 1000) or self.reward > self.grid_size*7
        self.wirecolors.append(self.connectors_dicc[self.this_connector][str(real)]['WireColor'])
        self.signal_groups.append(self.connectors_dicc[self.this_connector][str(real)]['signal_group'])
        self.already_pinned_thickness.append(connectors_dicc[self.this_connector][str(real)]['WireCSA'])
        self.already_pinned_in_out.append(connectors_dicc[self.this_connector][str(real)]['internal_pin'])
        self.multicore_pinned.append(connectors_dicc[self.this_connector][str(real)]['Multicore'])
        # Account for the boundaries of the grid
        self.agent_pos = self.agent_pos + 1
        self.terminal = bool(self.agent_pos > self.grid_size - 1)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return self.state, self.reward, self.terminal, info
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import TD3, PPO, A2C, SAC, DDPG
#from stable_baselines3.common.env_checker import check_env
from gym import logger
logger.set_level(40)
import time
dir="c:/temp/"

class agent2(gym.Env):
    # Custom Environment that follows gym interface.

    # observation space
    # 1 number pins [0,Num_Pin_unique_max]
    # 2 distance to center pin [0,11]bucket
    # 3 number of neighbors[0,Num_Pin_unique_max-1]
    # 4 internal/external[0,1]
    # 5 mean_neighbors [0,11]bucket
    # 6 distance centroide [0,11]bucket
    # 7 signal category [0,8]
    # 8 color wire categorical[0,max_color_categories]
    # 9 thickness wire continuous[0,51]bucket
    # 10 multicore yes/no [0,1]
    # 11 number of internal pins [0,Num_Pin_unique_max//2]
    # 12 nosignal [0,1]
    # 13 ismulticore [0,1]
    # 14 category multicore [0,max_multicore_categories]

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    # Define constants for clearer code

    def __init__(self, Num_Pin_unique_max, max_color_categories, max_categories_multicore, number_pins=100):

        super(agent2, self).__init__()
        # numero de pins connector -1 due to PartNumber
        # space number of pins of this connector
        self.lae = 14
        self.p = Num_Pin_unique_max
        self.c = max_color_categories
        self.m = max_categories_multicore
        self.action_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                       high=np.array(
                                           [self.p + 1, 11, self.p, 1, 11, 11, 9, self.c + 1, 51, 1, int(self.p // 2),
                                            1, 1, self.m + 1]),
                                       shape=(self.lae,))
        # observation space low/High bound number of diferent signals this connector
        # size, vector length number of pins
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                            high=np.array([self.p + 1, 11, self.p, 1, 11, 11, 9, self.c + 1, 51, 1,
                                                           int(self.p // 2), 1, 1, self.m + 1]),
                                            shape=(self.lae,))
        # estate list of actions
        self.agent_pos = 0
        self.bins10 = np.linspace(1, 10, 11)
        self.bins100 = np.linspace(0, 100, 101)
        self.binsp = np.linspace(0, self.p, self.p + 1)  # bin number of pins
        self.binsc = np.linspace(0, self.c, self.c + 1)  # bin number of
        self.binsm = np.linspace(0, self.m, self.m + 1)
        self.number_pins = number_pins
        self.signal_in_conector = []
        self.int_ext_in_conector = []
        self.color_in_conector = []
        self.number_pin_already_in_conector = []
        self.distances_in_conector = []
        self.no_signal_in_conector = []
        self.reward = 0
        return

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # np.zeros(self.lae, dtype='int')
        self.signal_in_conector = []
        self.int_ext_in_conector = []
        self.color_in_conector = []
        self.number_pin_already_in_conector = []
        self.distances_in_conector = []
        self.no_signal_in_conector = []
        self.agent_pos = 0
        self.reward = 0
        return self.state

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        #print(self.reward)
        return

    def get_reward_cat(self, signal_cat, int_ext, distance, no_signal_pin):
        # save the previous rewards
        reward = 0
        if signal_cat in self.signal_in_conector and signal_cat != 7:
            reward = reward - 1

        else:
            reward = reward + 4
        # higvoltage close in internal PIN or close to an empty cavity
        if (signal_cat in [1, 6,8]) and int_ext == 1:
            reward = reward + 5
        elif (signal_cat in [1, 6,8]) and no_signal_pin == 1:
            reward = reward + 2
        else:
            reward = reward - 5

        # if GROUND is present reward more when far from High Power categories
        if (signal_cat in [1, 6,8]) and 4 in self.signal_in_conector:
            reward = reward - int((distance + 10) // 4)
        else:
            reward = reward + int((distance + 10) // 4)

        if (signal_cat in [1, 6,8]) and 5 in self.signal_in_conector:
            reward = reward - int((distance + 10) // 4)
        else:
            reward = reward + int((distance + 10) // 4)
        # if High power     is present reward more when far from ground categories
        if (signal_cat in [4, 5,8]) and 1 in self.signal_in_conector:
            reward = reward - int((distance + 10) // 4)
        else:
            reward = reward + int((distance + 10) // 4)

        if (signal_cat in [4, 5,8]) and 6 in self.signal_in_conector:
            reward = reward - int((distance + 10) // 4)
        else:
            reward = reward + int((distance + 10) // 4)
        # reward more high power far from other signals
        if signal_cat == 6 and len(self.distances_in_conector) > 1:
            for ele in self.distances_in_conector:
                diff = abs(distance - ele)
                reward = reward + int((diff + 10) // 4)

        return reward

    def get_reward_color(self, color_cat):
        # save the previous rewards
        reward = 0
        if color_cat in self.color_in_conector:
            reward = reward - 4

        else:
            reward = reward + 4
        return reward

    def get_reward_int_ext(self, number_internal):
        # save the previous rewards
        reward = 0
        if number_internal > len(self.number_pin_already_in_conector):
            reward = reward - 4

        else:
            reward = reward + 4
        return reward

    def get_reward_multicore_out(self, distance, multicore):
        # save the previous rewards
        reward = 0
        if multicore == 1:
            reward = reward + int((distance + 10) // 2)

        else:
            reward = reward - int((distance + 10) // 2)
        return reward

    def get_bin_s(self, n, bins):
        _bin = 0
        for i in range(0, len(bins)):
            if bins[i] >= n:
                _bin = i - 1
                break
        return _bin

    def get_zero_one(self, n):
        if n < 0.1:
            return 0
        else:
            return 1

    def get_reward_empty_cavities(self, number_of_empty_cavities):
        reward = 0
        # if high power in connector reward more empty cavities
        if 6 in self.signal_in_conector or 1 in self.signal_in_conector:
            reward = int((10 + number_of_empty_cavities) // 10)
        elif number_of_empty_cavities < 2:
            reward = reward - 2

        return reward

    def calculate_reward(self, action):
        reward = 0
        distance = self.get_bin_s(action[1], self.bins10)  # distance to the center
        int_ext = self.get_zero_one(action[3])  # internal External Flag
        signal_cat = int(action[6])  # signal Category
        color_cat = int(action[7])  # color Category
        multicore = self.get_zero_one(action[9])  # multicore
        number_internal = int(action[10])  # number of internal pins
        no_signal_pin = self.get_zero_one(action[11])  # is an empty cavity
        number_of_empty_cavities = self.no_signal_in_conector.count(1)  # number of empty cavities

        # print(distance)
        reward = reward + self.get_reward_cat(signal_cat, int_ext, distance, no_signal_pin)
        reward = reward + self.get_reward_color(color_cat)
        reward = reward + self.get_reward_int_ext(number_internal)
        reward = reward + self.get_reward_multicore_out(distance, multicore)
        reward = reward + self.get_reward_empty_cavities(number_of_empty_cavities)
        # append what you see to the observation space
        self.signal_in_conector.append(signal_cat)
        self.int_ext_in_conector.append(int_ext)
        self.color_in_conector.append(color_cat)
        self.distances_in_conector.append(distance)
        self.no_signal_in_conector.append(no_signal_pin)
        if int_ext == 1:
            self.number_pin_already_in_conector.append(number_internal)

        return reward

    def step(self, action):

        # real=self.get_real(action) #  real values
        # calculate Reward

        self.reward = self.reward + self.calculate_reward(action)

        # Account for the boundaries of the grid
        self.agent_pos = self.agent_pos + 1
        self.terminal = bool(self.agent_pos > self.p)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return self.state, self.reward, self.terminal, info


def get_bin_10(n,bins10):
    """
    return bin for any arbitrary bins
    :param n:
    :param bins10:
    :return:
    """
    _bin=10
    for i in range(0,len(bins10)):
        if bins10[i]> n:
            _bin=1
            break
    return _bin


def get_pin_signal(sig_conn,i):
    """
    Get observation for prediction
    :param sig_conn:
    :param i:
    :return:
    """
    # create internal bins

    bins_distance= np.linspace(sig_conn['1']['min_distance'], sig_conn['1']['max_distance'], 11)
    bins_thickness= np.linspace(sig_conn['1']['Wire_WireCSA_min'], sig_conn['1']['Wire_WireCSA_max'], 51)
    # build observation
    Num_Pins=sig_conn[str(i)]['Num_Pins']
    distance_to_center=get_bin_10(sig_conn[str(i)]['distance_to_center'], bins_distance)
    neighbors_length=sig_conn[str(i)]['neighbors_length']
    internal_pin = 1 if sig_conn[str(i)]['internal_pin'] =='True' else 0
    mean_neighbors=get_bin_10(sig_conn[str(i)]['mean_neighbors'], bins_distance)
    centroide_distance=get_bin_10(sig_conn[str(i)]['centroide_distance'], bins_distance)
    signal_group=sig_conn[str(i)]['signal_group']
    Cat_Wire_WireColor=sig_conn[str(i)]['Cat_Wire_WireColor']
    WireCSA=get_bin_10(sig_conn[str(i)]['WireCSA'], bins_thickness)
    is_multicore = 0 if sig_conn[str(i)]['is_multicore'] =='na' else 1
    number_internal_pin= sig_conn[str(i)]['number_internal_pin']
    nosignal = sig_conn[str(i)]['no_signal_pin']
    ismulticore= sig_conn[str(i)]['is_multicore']
    cat_multicore=sig_conn[str(i)]['multicore_same']
    return list([Num_Pins,distance_to_center,neighbors_length,internal_pin,mean_neighbors,
                 centroide_distance,signal_group,Cat_Wire_WireColor,WireCSA,is_multicore,
                 number_internal_pin,nosignal,ismulticore, cat_multicore])


def train_agent2(connectors_dicc, connector, data, max_categories, algo, size):
    """
    Train agent 2
    :param Num_Pin_unique_max:
    :param max_color_categories:
    :param max_categories_multicore:
    :param Num_Pins:
    :return:
    """
    Num_Pin_unique_max =  connectors_dicc[connector]['1']['Num_Pin_unique_max']
    max_color_categories =  connectors_dicc[connector]['1']['Cat_Wire_WireColor_max']
    max_categories_multicore = connectors_dicc[connector]['1']['max_categories_multicore']
    Num_Pins = connectors_dicc[connector]['1']['Num_Pins']
    dicc_signals = eval(connectors_dicc[connector]['1']['dicc_signals'])
    num_signals = len(dicc_signals.keys())

    if num_signals < 10 and Num_Pins < 10 :
        number_possible_variations = num_signals ** Num_Pins
    else:
        number_possible_variations =50000
    num_iterations = 0
    #power = len(str(number_possible_variations)) - 5
    if number_possible_variations < 20000:
        num_iterations = 20000
    else:

        num_iterations = 50000

    env_train = DummyVecEnv(
        [lambda: agent2(Num_Pin_unique_max, max_color_categories, max_categories_multicore, Num_Pins)])

    #create model
    if algo == "TD3":
        model = TD3('MlpPolicy', env_train, verbose=2, buffer_size=5000,create_eval_env=True)
    elif algo == "PPO":
        model = PPO('MlpPolicy', env_train, verbose=2, create_eval_env=True)
    elif algo == "A2C":
        model = A2C('MlpPolicy', env_train, verbose=2, create_eval_env=True)
    elif algo == "SAC":
        model = SAC('MlpPolicy', env_train, verbose=2, create_eval_env=True)
    elif algo == "DDPG":
        model = DDPG('MlpPolicy', env_train, verbose=2, create_eval_env=True)

    time1 = time.time()
    model.learn(num_iterations)
    time2 = time.time()

    model_name=f"{algo}_agent2"
    model.save(dir+ f"models/model_{model_name}_{size}.pkl")
    print(f"Saved Model Agent2 Size {size} training time {time2 - time1}")
    return


def predict_agent2(connectors_dicc, connector, algo):

    Num_Pin_unique_max = connectors_dicc[connector]['1']['Num_Pin_unique_max']
    max_color_categories = connectors_dicc[connector]['1']['Cat_Wire_WireColor_max']
    max_categories_multicore = connectors_dicc[connector]['1']['max_categories_multicore']
    Num_Pins = connectors_dicc[connector]['1']['Num_Pins']

    env = agent2(Num_Pin_unique_max, max_color_categories, max_categories_multicore, Num_Pins)

    # env.render()
    # check_env(env, warn=True)
    # print(env.observation_space)
    # print(env.action_space)
    # print(env.action_space.sample())
    if algo == "TD3":
        model_name = f"{algo}_agent2"
        model = TD3.load(dir+ f"models/model_{model_name}.pkl")
    elif algo == "PPO":
        model_name = f"{algo}_agent2"
        model = PPO.load(dir+ f"models/model_{model_name}.pkl")
    elif algo == "A2C":
        model_name = f"{algo}_agent2"
        model = A2C.load(dir+ f"models/model_{model_name}.pkl")
    elif algo == "SAC":
        model_name = f"{algo}_agent2"
        model = SAC.load(dir+ f"models/model_{model_name}.pkl")
    elif algo == "DDPG":
        model_name = f"{algo}_agent2"
        model = DDPG.load(dir+ f"models/model_{model_name}.pkl")

    obs = env.reset()
    reward=0
    for key in connectors_dicc[connector].keys():
        if key != 'PartNumber':
            obs = np.array(get_pin_signal(connectors_dicc[connector], key))
            action, _state = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action.astype(int))

            env.render()
            if done:
                obs = env.reset()

    return reward


def demo_agent2(connectors_dicc, connector, algo):
    """

    :param connectors_dicc:
    :param connector:
    :param algo:
    :return:
    """
    Num_Pin_unique_max = connectors_dicc[connector]['1']['Num_Pin_unique_max']
    max_color_categories = connectors_dicc[connector]['1']['Cat_Wire_WireColor_max']
    max_categories_multicore = connectors_dicc[connector]['1']['max_categories_multicore']
    Num_Pins = connectors_dicc[connector]['1']['Num_Pins']

    env = agent2(Num_Pin_unique_max, max_color_categories, max_categories_multicore, Num_Pins)

    # env.render()
    # check_env(env, warn=True)
    # print(env.observation_space)
    # print(env.action_space)
    # print(env.action_space.sample())
    if algo == "TD3":
        model_name = f"{algo}_agent2"
        model = TD3.load(dir+ f"models/model_{model_name}_{Num_Pins}.pkl")
    elif algo == "PPO":
        model_name = f"{algo}_agent2"
        model = PPO.load(dir+ f"models/model_{model_name}_{Num_Pins}.pkl")
    elif algo == "A2C":
        model_name = f"{algo}_agent2"
        model = A2C.load(dir+ f"models/model_{model_name}_{Num_Pins}.pkl")
    elif algo == "SAC":
        model_name = f"{algo}_agent2"
        model = SAC.load(dir+ f"models/model_{model_name}_{Num_Pins}.pkl")
    elif algo == "DDPG":
        model_name = f"{algo}_agent2"
        model = DDPG.load(dir+ f"models/model_{model_name}_{Num_Pins}.pkl")


    return model, env


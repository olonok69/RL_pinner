from pre_processing import *
from IPython import embed
from connector import *
from connector2 import *
from utils import *
import argparse
from stable_baselines3.common.env_checker import check_env



def main():
   parser = argparse.ArgumentParser("Agent Pinner")

   parser.add_argument("--mode", type=str, help="training or prediction", required=True)
   parser.add_argument("--connector", type=str, help="TO_111_LH_F/143_3P", required=True)
   parser.add_argument("--agent", type=str, help="1-> pin specialized 2-> generic", required=True)
   parser.add_argument("--algo", type=str, help="1-->DQN,2-->TD3, PPO, A2C, SAC, DDPG ", required=True)

   args = parser.parse_args()
   if args.mode == 'training':

      print("Preprocessing data")
      data1 = pd.read_csv("data/symbol_pins.csv")
      symbols=get_neighbors_partnumbers(data1)
      #symbols = get_pin_int_out(data1, symbols)

      data = pd.read_csv("data/inline_pins.csv")
      # prepocressing files and
      data, d, max_categories = preprocess_data(data, symbols)

      print("Creating Dicctionary for Training")
      connectors_dicc=create_connector_dicc(data)
      # save dictionary
      path = "conf"
      file = "connectors_dicc.pkl"
      save_dict_pickle(path, connectors_dicc, file)
      path = "conf"
      file = "signal_cat.pkl"
      save_dict_pickle(path, d, file)

      data.to_pickle("data/connector_processed.pickle")
   elif args.mode == 'prediction':
      path = "conf"
      file = "connectors_dicc.pkl"
      connectors_dicc = load_dicc(path, file)
      path = "conf"
      file = "signal_cat.pkl"
      d = load_dicc(path, file)
      data= pd.read_pickle("data/connector_processed.pickle")

   # observation space is a vector length number of pins. each element is an int to choose among the list of
   # signal categories

   connector = args.connector

   if args.mode == 'training' and args.agent == "1" and args.algo =="DQN":
      print("Training....")
      # train DQN agent --> agent spezialized in connectors with same number of pins
      model = train_agent1(connectors_dicc, connector, data, max_categories)
      return
   elif args.mode == 'training' and args.agent == "2" and  args.algo !="DQN":
      algo = args.algo
      if algo in ["TD3", "PPO", "A2C", "SAC", "DDPG"]:
       model = train_agent2(connectors_dicc, connector, data, max_categories, algo)
      return

   elif args.mode == 'prediction' and args.agent == "1" and args.algo =="DQN":


      # number of pins this connector
      len_test_conn = len(connectors_dicc[connector].keys()) - 1
      # list signals this connector to evaluate
      data1 = data[data['Connector Name'] == connector]
      list_signals = list(data1['Cat_Signal Name'].unique())
      # state space
      state_space = len(eval(connectors_dicc[connector]['1']['dicc_signals']).keys())
      # max actegories
      max_categories = (len(eval(connectors_dicc[connector]['1']['dicc_signals']).keys()))
      signals_cats = []
      for key in connectors_dicc[connector].keys():
         if key != "PartNumber":
            signals_cats.append(connectors_dicc[connector][key]['Cat_Signal'])



      reward = predict_agent1(connectors_dicc, connector, list_signals, max_categories, state_space, signals_cats)

      print(f"Reward Agent1 for connector {connector}, with number pins {len_test_conn} : {reward}")

   elif args.mode == 'prediction' and args.agent == "2" and args.algo !="DQN":
      algo = args.algo
      if algo in ["TD3", "PPO", "A2C", "SAC", "DDPG"]:
         # number of pins this connector
         len_test_conn = len(connectors_dicc[connector].keys()) - 1
         reward = predict_agent2(connectors_dicc, connector, algo)
         print(f"Reward Agent2 for connector {connector}, with number pins {len_test_conn} : {reward}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

#

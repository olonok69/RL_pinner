from pre_processing import *

from connector import *
import sys
from utils import *
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

dir="c:/temp/"

def main():
   parser = argparse.ArgumentParser("Agent Pinner")

   parser.add_argument("--mode", type=str, help="training , prediction, demo", required=True)
   parser.add_argument("--connector", type=str, help="TO_111_LH_F/143_3P", required=True)
   parser.add_argument("--agent", type=str, help="1-> pin specialized 2-> generic", required=True)
   parser.add_argument("--algo", type=str, help="1-->DQN,2-->TD3, PPO, A2C, SAC, DDPG ", required=True)

   args = parser.parse_args()
   # create directories
   if not os.path.exists(dir):
      os.makedirs(dir)
   if not os.path.exists(dir+"data"):
      os.makedirs(dir+"data")
   if not os.path.exists(dir+"conf"):
      os.makedirs(dir+"conf")
   if not os.path.exists(dir+"models"):
      os.makedirs(dir+"models")

   if args.mode == 'training':

      print("Preprocessing data")
      data1 = pd.read_csv(dir+"data/symbol_pins.csv")
      symbols=get_neighbors_partnumbers(data1)
      #symbols = get_pin_int_out(data1, symbols)

      data = pd.read_csv(dir+"data/inline_pins.csv")
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

      data.to_pickle(dir+"data/connector_processed.pickle")
      symbols.to_pickle(dir+"data/symbols_processed.pickle")
   elif args.mode == 'prediction' or args.mode == 'demo':

      # prediction
      move=0
      symbols= pd.read_pickle(dir+"data/symbols_processed.pickle")
      example=pd.read_csv(dir+"data/new_connector.csv")
      num_part_number=example[['Connector PartNumber']].isnull().sum().sum()
      if num_part_number ==0:
         move=1
         data, d, max_categories = preprocess_data(example, symbols)
         connectors_dicc = create_connector_dicc(data)


   # observation space is a vector length number of pins. each element is an int to choose among the list of
   # signal categories


   if args.mode == 'training' and args.agent == "1" and args.algo =="DQN":
      connector = args.connector
      print("Training....")
      # train DQN agent --> agent spezialized in connectors with same number of pins
      model = train_agent1(connectors_dicc, connector, data, max_categories)
      return
   elif args.mode == 'training' and args.agent == "2" and  args.algo !="DQN":
      time1 = time.time()
      d_models, model_connectors = get_models_sizes(connectors_dicc)

      connector = args.connector
      algo = args.algo
      if algo in ["TD3", "PPO", "A2C", "SAC", "DDPG"]:
         # train a model for each size
         status= train_models_multi(d_models, connectors_dicc,  data, max_categories, algo, model_connectors)


      time2 = time.time()
      print(f"total time: {time2 - time1}")
      return

   elif args.mode == 'prediction' and args.agent == "1" and args.algo =="DQN":

      connector = list(data['Connector Name'].unique())[0]
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

   elif args.mode == 'prediction' and args.agent == "2" and args.algo !="DQN" and move ==1:
      time1= time.time()
      connector = list(data['Connector Name'].unique())[0]
      algo = args.algo
      Num_Pins = connectors_dicc[connector]['1']['Num_Pins']
      dicc_signals = eval(connectors_dicc[connector]['1']['dicc_signals'])
      num_signals = len(dicc_signals.keys())
      data ,number_possible_variations, num_iterations = create_dataframe_test_2(Num_Pins, num_signals, dicc_signals)

      max_reward = calculate_max_reward(Num_Pins,  connectors_dicc, connector)
      print(max_reward)
      if algo in ["TD3", "PPO", "A2C", "SAC", "DDPG"]:
         # number of pins this connector
         # len_test_conn = len(connectors_dicc[connector].keys()) - 1
         # reward = predict_agent2(connectors_dicc, connector, algo)
         # print(f"Reward Agent2 for connector {connector}, with number pins {len_test_conn} : {reward}")
         # data['reward']=reward
         # data.to_csv(dir+"data/new_connector_pred.csv")
         data.to_csv(dir + "data/prediction_pins.csv", index=False)
         data_with_predictions = prediction_multiprocessing(data, algo, connectors_dicc, connector)
         data_with_predictions.to_csv(dir + "data/prediction_pins_with_p.csv", index=False)
         dt = data_with_predictions.sort_values("prediction", ascending=False)
         dt['max_reward']= max_reward
         dt['number_possible_variations'] = number_possible_variations
         dt['num_iterations'] = num_iterations
         dt[:1].to_csv(dir + "data/best_prediction.csv", index=False)
      time2=time.time()
      print(f"total time: {time2-time1}")

   elif args.mode == 'prediction' and args.agent == "2" and args.algo != "DQN" and move == 0:
      print("We can not continue. We miss important attributes to build model")
      sys.exit(2)

   elif args.mode == 'demo' and args.agent == "2" and args.algo != "DQN":
       connector = list(data['Connector Name'].unique())[0]
       algo = args.algo
       num_signals = len(dicc_13.keys())
       Num_Pins = connectors_dicc[connector]['1']['Num_Pins']
       data= create_dataframe_test(Num_Pins, num_signals)
       data.to_csv(dir+"data/demo_13_pins.csv", index=False)
       #model, env = demo_agent2(connectors_dicc, connector, algo)

       data_with_predictions = predict_demo(data, algo, connectors_dicc, connector)
       data_with_predictions.to_csv(dir+"data/demo_13_pins_with_predictions.csv", index=False)







   # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

#

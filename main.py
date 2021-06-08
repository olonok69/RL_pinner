import pandas as pd
from pre_processing import *
from IPython import embed
from stable_baselines3.common.vec_env import DummyVecEnv
from connector import *



def main():
   data1 = pd.read_csv("data/symbol_pins.csv")
   symbols=get_neighbors_partnumbers(data1)
   #symbols = get_pin_int_out(data1, symbols)


   data = pd.read_csv("data/inline_pins.csv")
   data, d, max_categories = preprocess_data(data, symbols)

   connectors_dicc=create_connector_dicc(data)
   # observation space is a vector length number of pins. each element is an int to choose among the list of
   # signal categories
   connector = "TO_169_F/161_RH_13P"
   len_test_conn=len(connectors_dicc[connector].keys()) - 1
   data1 = data[data['Connector Name'] == connector]

   list_signals = list(data1['Cat_Signal Name'].unique())
   # state space is all the category signal we show for that kind of connector with n pins . calculated in
   # pre_processing dictionary dicc__signals
   state_space = len(eval(connectors_dicc[connector]['1']['dicc_signals']).keys())


   embed()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

#

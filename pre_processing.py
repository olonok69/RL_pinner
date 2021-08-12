import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import random
from connector2 import demo_agent2
import numpy as np
import copy
import pickle
import os
from math import sqrt
import multiprocessing
import functools
from connector2 import *



dicc_13={0: 'CLEARANCE_SNR_ECU_0885', 1: 'CLEARANCE_SNR_ECU_0890', 2: 'EQ_FUEL_PRESS_SSR_HI-TNGA_E2',
         3: 'EQ_FUEL_PRESS_SSR_HI-TNGA_PR', 4: 'EQ_FUEL_PRESS_SSR_HI-TNGA_VC', 5: 'EQ_NE_SSR-TNGA_NE-',
         6: 'EQ_NE_SSR-TNGA_NEP', 7: 'EQ_NE_SSR-TNGA_VCNE', 8: 'EQ_PDB_L_1', 9: 'EQ_PDB_L_2', 10: 'EQ_PDB_L_5',
         11: 'GND_BK_UP_LP_LH_E', 12: 'HV_ECU-HV_0024', 13: 'HV_ECU-HV_0025', 14: 'SLD_EFI_ECU-TNGA_KNOCK_SSR_101',
         15: 'SLD_EFI_ECU-TNGA_KNOCK_SSR_102', 16: 'SLD_MG_ECU_197', 17: 'SLD_MG_ECU_198', 18: 'SLD_MG_ECU_199',
         19: 'SLD_MG_ECU_200', 20: 'SLD_MG_ECU_201', 21: 'SLD_MG_ECU_202', 22: 'SLD_MG_ECU_203',
         23: 'TW_CLEARANCE_SNR_ECU_055', 24: 'TW_CLEARANCE_SNR_ECU_056', 25: 'TW_CLEARANCE_SNR_ECU_061',
         26: 'TW_CLEARANCE_SNR_ECU_062', 27: 'na'}

def cal_distance_center(row):
    """

    :param row:
    :return:
    """

    xs= 0-float(row["Pin CenterX"])
    ys= 0-float(row["Pin CenterY"])
    return sqrt((xs*xs)+(ys*ys))

def get_neighbors(row, distances):
    """

    :param row:
    :param distances:
    :return:
    """
    pin_row=row["Pin Name"]
    distances_sorted=sorted(distances[pin_row], key=distances[pin_row].get, reverse=False)
    ts=distances[pin_row][distances_sorted[0]]*1.90
    return [x for x in  distances_sorted if distances[pin_row][x] < ts]

def get_neighbors_length(row):
    n=row['neighbors']
    return len(n)

def get_list_empty(row):
    return ['0']


def myfunc_data(data):
    """

    :param data:
    :return:
    """
    # function groupping by
    data['number_internal_pin'] = len(data[data["internal_pin"] == "True"])

    return data

def get_neighbors_partnumbers(data):
    """

    :param data:
    :return:
    """
    data["distance_to_center"] = data.apply(lambda row: cal_distance_center(row), axis=1)
    list_pn = list(data['Symbol Name'].unique())
    cols = ['Symbol Name', 'Pin Name', 'Pin CenterY', 'Pin CenterX', 'Pin Width',
            'Pin Height', 'distance_to_center', 'neighbors','neighbors_length']

    dataout = pd.DataFrame([], columns=cols)
    for pn in list_pn:
        data1 = data[data['Symbol Name'] == pn]
        if len(data1) > 1:

            data_json = data1.to_json(orient="records")
            dicc_json1 = eval(data_json)
            dicc_json2 = eval(data_json)
            distances = {}
            for ele in dicc_json1:
                pin = ele['Pin Name']
                distances[pin] = {}
                x1 = ele['Pin CenterX']
                y1 = ele['Pin CenterY']
                for ele2 in dicc_json2:
                    pin2 = ele2['Pin Name']
                    if ele2 != ele:
                        x2 = ele2['Pin CenterX']
                        y2 = ele2['Pin CenterY']
                        distance = sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)))
                        distances[pin][pin2] = distance
            #  get neighbors
            data1['neighbors'] = data1.apply(lambda row: get_neighbors(row, distances), axis=1)
            data1['neighbors_length'] = data1.apply(lambda row: get_neighbors_length(row), axis=1)
            dataout = pd.concat([dataout, data1])
        else:
            #print(pn)
            data1['neighbors'] = data1.apply(lambda row: get_list_empty(row), axis=1)
            data1['neighbors_length'] = data1.apply(lambda row: get_neighbors_length(row), axis=1)
            dataout = pd.concat([dataout, data1])

    # New fiellds
    dataout["internal_pin"] = ""
    dataout["mean_neighbors"] = ""
    dataout["centroide_distance"] = ""
    cols = ['Symbol Name', 'Pin Name', 'Pin CenterY', 'Pin CenterX', 'Pin Width',
            'Pin Height', 'distance_to_center', 'neighbors', 'neighbors_length', 'internal_pin',
            'mean_neighbors', 'centroide_distance']

    dataout2 = pd.DataFrame([], columns=cols)
    for pn in list_pn:
        data1 = dataout[dataout['Symbol Name']==pn]
        if len(data1) > 4:
            for i, row in data1.iterrows():
                data_json= data1.to_json(orient="records")
                #dicc_json1=eval(data_json)
                dicc_json2=eval(data_json)
                ndistances=[]
                xs=[]
                ys=[]

                pin=data1.at[i,'Pin Name']

                distance_to_center=data1.at[i,'distance_to_center']
                neighbors=data1.at[i,"neighbors"]

                for ele2 in dicc_json2:
                    pin2=ele2['Pin Name']

                    if pin2 in neighbors:
                        ndistances.append(ele2['distance_to_center'])
                        xs.append(ele2['Pin CenterX'])
                        ys.append(ele2['Pin CenterY'])


                    ndistances=sorted(ndistances)
                nave= np.mean(ndistances)
                xg=0-(np.sum(xs)/len(xs))
                yg=0-(np.sum(ys)/len(ys))
                #calculate distance centroide to center
                centroide_distance=sqrt((xg*xg)+(yg*yg))
                #print(f"{centroide_distance} {distance_to_center}")
                data1.at[i,'internal_pin']= str((centroide_distance*1.1> distance_to_center>centroide_distance*0.90)
                                                and (distance_to_center>centroide_distance))
                data1.at[i, 'mean_neighbors'] = nave
                data1.at[i, 'centroide_distance'] = centroide_distance
            dataout2 = pd.concat([dataout2, data1])
        else:
            #print(pn)
            data1['internal_pin']=str(False)
            data1['mean_neighbors'] = 0
            data1['centroide_distance'] = 0
            dataout2 = pd.concat([dataout2, data1])

    dataout2 = dataout2.groupby('Symbol Name').apply(myfunc_data)
    groupped=dataout2.groupby("Symbol Name").agg({"distance_to_center":["min","max"] })
    groupped.reset_index(inplace=True)
    groupped.columns=['Symbol Name', 'min_distance', 'max_distance']

    final = pd.merge(dataout2, groupped, on='Symbol Name', how='inner')
    return final


def signal_equal(row):
    if (row["Pin PreferredSignal"]== "na" and row["Signal Name"]== "na") :
        return 2
    elif row["Pin PreferredSignal"]==row["Signal Name"]:
        return 1
    else:
        return 0

def get_key(x,col):
    """
    get the key from a dictionary and apply to column
    :param x:
    :param d:
    :param col:
    :return:
    """
    d = eval(x['dicc_signals'])
    key_list = list(d.keys())
    val_list = list(d.values())
    try:
        ind = val_list.index(x[col])
        return key_list[ind]
    except:
        return len(val_list) + 1

def category_signal(row):
    """

    :param row:
    :return:
    """
    if row["Signal Name"][:5]=='BATT_':
        return 1
    elif row["Signal Name"][:4]=='CAN_':
        return 2
    elif row["Signal Name"][:6]=='CANFD_':
        return 3
    elif row["Signal Name"][:4]=='GND_' or 'GND' in row["Signal Name"] :
        return 4
    elif row["Signal Name"][:9]=='GNDR_GND_':
        return 5
    elif row["Signal Name"][:3]=='HV_' or 'POWER' in row['Signal Name']:
        return 6
    elif row["Signal Name"] =='na':
        return 8
    else:
        return 7

def category_signal2(row):
    """

    :param row:
    :return:
    """
    if row["Signal Name"][-3:]=='_HI' or row["Signal Name"][-3:]=='_LO':
        return 1
    else:
        return 0

def cat_pergroup_signals(data):
    """
    apply category to a group of signals
    :param data:
    :return:
    """
    data['Cat_Signal Name'] = data["Signal Name"].astype('category').cat.codes

    return data

def get_dictionary(data):
    """
    get dictionary
    :param data:
    :return:
    """
    c = data["Signal Name"].astype('category')
    data['dicc_signals'] = str(dict(enumerate(c.cat.categories)))
    return data

def no_signal_in_pin(row):
    """
    No signal in that pin
    :param row:
    :return:
    """
    if row["Signal Name"]=="na":
        return 1
    else:
        return 0

def multicore_data(data):
    """
    grouuping categories per connector
    :param data:
    :return:
    """
    data['multicore_same'] = data["MulticoreInnerToOutter1"].astype('category').cat.codes

    return data

def is_multicore(row):
    """
    cable is part of a multicore
    :param row:
    :return:
    """
    if row['MulticoreInnerToOutter1']!="na":
        return 1
    else:
        return 0

def preprocess_data(data, symbols):
    """

    :param data:
    :return:
    """
    #fill Nan
    data["Signal Name"].fillna("na", inplace=True) # Signal name
    data["Pin PreferredSignal"].fillna("na", inplace=True) #Pin Preferredd Signal
    data['Connector PartNumber'].fillna("NOPARTNUMBER", inplace=True) # Part Number connector
    data['Wire WireColor'].fillna("na", inplace=True) #color
    data['Wire WireCSA'].fillna(0, inplace=True) # thickness
    data['MulticoreInnerToOutter1'].fillna("na", inplace=True)# Multicore
    # check signal method
    data['check_signal'] = data.apply(lambda row: signal_equal(row), axis=1)
    # count number of different signals per connector
    connector_view = data.groupby(["Connector Name", 'Pin Name']).agg({"Pin PreferredSignal": ['nunique', 'count']})
    levels = connector_view.index.nlevels
    for i in reversed(range(0, levels)):
        connector_view.reset_index(level=i, inplace=True)

    connector_view.columns = ['Connector Name', 'Pin Name', 'Pin Name nunique', "Pin Name count"]
    # join the calculated columns
    groupped = pd.merge(connector_view, data, on=['Connector Name', 'Pin Name'], how='inner')
    # count number of pins per connector
    connector_view2 = data.groupby(["Connector Name"]).agg({'Pin Name': ['nunique', 'count']})
    levels = connector_view2.index.nlevels
    for i in reversed(range(0, levels)):
        connector_view2.reset_index(level=i, inplace=True)
    connector_view2.columns = ['Connector Name', 'Num_Pin_unique', 'Num_Pins']
    # check if a pin have multiple signals
    connector_view2['more_signal_in_pin'] = connector_view2["Num_Pin_unique"] != connector_view2["Num_Pins"]
    # join info to dataframe
    groupped2 = pd.merge(groupped, connector_view2, on=['Connector Name'], how='inner')
    groupped2['Connector PartNumber'].fillna("NOPARTNUMBER", inplace=True)
    # category group according to Prefix
    groupped2["signal_cat"] = groupped2.apply(lambda row: category_signal(row), axis=1)
    # category group according to Sufix
    groupped2["signal_multicore"] = groupped2.apply(lambda row: category_signal2(row), axis=1)
    #apply categoryzation per connector
    groupped2 = groupped2.groupby("Num_Pins").apply(cat_pergroup_signals)
    #getdictionary into a column
    groupped2 = groupped2.groupby("Num_Pins").apply(get_dictionary)

    #groupped2['Cat_Signal Name'] = groupped2["Signal Name"].astype('category').cat.codes
    #still interesting knows all signals on the car
    c = groupped2["Signal Name"].astype('category')
    d = dict(enumerate(c.cat.categories))
    # encode "Pin PreferredSignal"
    groupped2['Cat_PreferredSignal'] = groupped2.apply(lambda row: get_key(row, "Pin PreferredSignal"), axis=1)
    #number of different signals
    num_categories= len(d.keys())
    # Color categories and min max thickness
    groupped2["Cat_Wire_WireColor"] = groupped2["Wire WireColor"].astype('category').cat.codes
    groupped2["Cat_Wire_WireColor_max"] = groupped2["Cat_Wire_WireColor"].max()
    groupped2["Wire_WireCSA_min"] = groupped2["Wire WireCSA"].min()
    groupped2["Wire_WireCSA_max"] = groupped2["Wire WireCSA"].max()
    groupped2['Num_Pin_unique_max'] = groupped2['Num_Pin_unique'].max()
    # there is no signal mapped to this pin
    groupped2["no_signal_pin"] = groupped2.apply(lambda row: no_signal_in_pin(row), axis=1)
    # group of multicores in this  connector , including the na
    groupped2 = groupped2.groupby('Connector Name').apply(multicore_data)
    # signal is part of a multicore
    groupped2['is_multicore'] = groupped2.apply(lambda row: is_multicore(row), axis=1)
    # max category multicore
    groupped2['max_categories_multicore'] = groupped2['multicore_same'].max()

    #join symbols information, distances and neighbors
    groupped2['Pin Name'] = groupped2['Pin Name'].astype(str)
    groupped2 = pd.merge(groupped2, symbols, left_on=['Connector PartNumber', 'Pin Name'],
                         right_on=['Symbol Name', 'Pin Name'], how="left")

    return groupped2, d,num_categories

def create_connector_dicc(groupped2):
    """

    :param data:
    :return:
    """
    connectors = {}
    conn_to_save = ""
    for i, row in groupped2.iterrows():
        connector = groupped2.loc[i, "Connector Name"]
        pin = groupped2.loc[i, 'Pin Name']

        if conn_to_save == "" or groupped2.loc[i - 1, "Connector Name"] == connector:
            try:

                connectors[connector][pin] = {}
            except:
                connectors[connector] = {}
                connectors[connector][pin] = {}

            conn_to_save = "NO"
        else:
            conn_to_save = ""
            try:

                connectors[connector][pin] = {}
            except:
                connectors[connector] = {}
                connectors[connector][pin] = {}

        # Part Number at level of PIN
        PartNumber = groupped2.loc[i, 'Connector PartNumber']
        connectors[connector]["PartNumber"] = PartNumber
        # prefered Signal
        PreferredSignal = groupped2.loc[i, 'Pin PreferredSignal']
        connectors[connector][pin]['PreferredSignal'] = PreferredSignal
        Cat_PreferredSignal = groupped2.loc[i, 'Cat_PreferredSignal']
        connectors[connector][pin]['Cat_PreferredSignal'] = Cat_PreferredSignal

        # Signal in that pin
        Signal = groupped2.loc[i, 'Signal Name']
        connectors[connector][pin]['Signal'] = Signal
        Cat_Signal = groupped2.loc[i, 'Cat_Signal Name']
        connectors[connector][pin]['Cat_Signal'] = Cat_Signal
        # dictionary of categories
        dicc_signals = groupped2.loc[i, 'dicc_signals']
        connectors[connector][pin]['dicc_signals'] = dicc_signals

        # Check Signal 1 preferred = Signal, 0 1 preferred != Signal, 2 both NaN
        check_signal = groupped2.loc[i, 'check_signal']
        connectors[connector][pin]['check_signal'] = check_signal
        # Number of pins connector
        Num_Pins = groupped2.loc[i, 'Num_Pins']
        connectors[connector][pin]['Num_Pins'] = Num_Pins
        # Number of unique Pins in this connector
        Num_Pin_unique = groupped2.loc[i, 'Num_Pin_unique']
        connectors[connector][pin]['Num_Pin_unique'] = Num_Pin_unique
        # True if number of Pins is not the same of number of unique Pins
        more_signal_in_pin = groupped2.loc[i, 'more_signal_in_pin']
        connectors[connector][pin]['more_signal_in_pin'] = more_signal_in_pin

        # WireColor
        WireColor = groupped2.loc[i, 'Wire WireColor']
        connectors[connector][pin]['WireColor'] = WireColor
        # WireCSA
        WireCSA = groupped2.loc[i, 'Wire WireCSA']
        connectors[connector][pin]['WireCSA'] = WireCSA
        # WireCSA
        Multicore = groupped2.loc[i, 'MulticoreInnerToOutter1']
        connectors[connector][pin]['Multicore'] = Multicore

        # signal_category Group
        signal_group = groupped2.loc[i, 'signal_cat']
        connectors[connector][pin]['signal_group'] = signal_group

        # internal connector / external  connertor  internal_pin
        internal_pin = groupped2.loc[i, 'internal_pin']
        connectors[connector][pin]['internal_pin'] = internal_pin

        # number of neigbourgs
        neighbors_length = groupped2.loc[i, 'neighbors_length']
        connectors[connector][pin]['neighbors_length'] = neighbors_length

        # mean_neighbors
        mean_neighbors = groupped2.loc[i, 'mean_neighbors']
        connectors[connector][pin]['mean_neighbors'] = mean_neighbors
        # centroide_distance
        centroide_distance = groupped2.loc[i, 'centroide_distance']
        connectors[connector][pin]['centroide_distance'] = centroide_distance

        # number_internal_pin
        number_internal_pin = groupped2.loc[i, 'number_internal_pin']
        connectors[connector][pin]['number_internal_pin'] = number_internal_pin
        # max_distance
        max_distance = groupped2.loc[i, 'max_distance']
        connectors[connector][pin]['max_distance'] = max_distance
        # max_distance
        min_distance = groupped2.loc[i, 'min_distance']
        connectors[connector][pin]['min_distance'] = min_distance
        # Cat_Wire_WireColor
        Cat_Wire_WireColor = groupped2.loc[i, 'Cat_Wire_WireColor']
        connectors[connector][pin]['Cat_Wire_WireColor'] = Cat_Wire_WireColor
        # Cat_Wire_WireColor
        Cat_Wire_WireColor_max = groupped2.loc[i, 'Cat_Wire_WireColor_max']
        connectors[connector][pin]['Cat_Wire_WireColor_max'] = Cat_Wire_WireColor_max
        # Wire_WireCSA_min
        Wire_WireCSA_min = groupped2.loc[i, 'Wire_WireCSA_min']
        connectors[connector][pin]['Wire_WireCSA_min'] = Wire_WireCSA_min
        # Wire_WireCSA_max
        Wire_WireCSA_max = groupped2.loc[i, 'Wire_WireCSA_max']
        connectors[connector][pin]['Wire_WireCSA_max'] = Wire_WireCSA_max
        # Num_Pin_unique_max
        Num_Pin_unique_max = groupped2.loc[i, 'Num_Pin_unique_max']
        connectors[connector][pin]['Num_Pin_unique_max'] = Num_Pin_unique_max

        # no_signal_pin
        no_signal_pin = groupped2.loc[i, 'no_signal_pin']
        connectors[connector][pin]['no_signal_pin'] = no_signal_pin
        # multicore_same detected 8 categories
        multicore_same = groupped2.loc[i, 'multicore_same']
        connectors[connector][pin]['multicore_same'] = multicore_same
        # is_multicore
        is_multicore = groupped2.loc[i, 'is_multicore']
        connectors[connector][pin]['is_multicore'] = is_multicore
        # max_categories_multicore
        max_categories_multicore = groupped2.loc[i, 'max_categories_multicore']
        connectors[connector][pin]['max_categories_multicore'] = max_categories_multicore

        # signal_category Group
        try:
            distance_to_center = groupped2.loc[i, 'distance_to_center']
            neighbors = groupped2.loc[i, 'neighbors']
            connectors[connector][pin]['distance_to_center'] = distance_to_center
            connectors[connector][pin]['neighbors'] = neighbors
        except:
            connectors[connector][pin]['distance_to_center'] = 0
            connectors[connector][pin]['neighbors'] = []

    return connectors

def save_dict_pickle(dict, file, path):
    """

    :param dict:
    :param file:
    :param path:
    :return:
    """
    modelpath = os.path.join(path, file)
    with open(modelpath, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def get_bin_10(n,bins10):
    """
    get bin index
    :param n:
    :param bins10:
    :return:
    """
    _bin=10
    for i in range(0,len(bins10)):
        if bins10[i]> n:
            _bin=i-1
            break
    return _bin

def get_pin_signal(sig_conn,i):
    """
    #1 number pins [0,Num_Pin_unique_max]
    #2 distance to center pin [0,11]bucket
    #3 number of neighbors[0,Num_Pin_unique_max-1]
    #4 internal/external[0,1]
    #5 mean_neighbors [0,11]bucket
    #6 distance centroide [0,11]bucket
    #7 signal category [0,7]
    #8 color wire categorical[0,max_color_categories]
    #9 thickness wire continuous[0,51]bucket
    #10 multicore yes/no [0,1]
    #11 number of internal pins [0,Num_Pin_unique_max//2]
    #12 nosignal [0,1]
    #13 ismulticore [0,1]
    #14 category multicore [0,max_multicore_categories]
    get observation from File
    :param sig_conn:
    :param i:
    :return:
    """
    # create bins
    bins_distance= np.linspace(sig_conn['1']['min_distance'], sig_conn['1']['max_distance'], 11)
    bins_thickness= np.linspace(sig_conn['1']['Wire_WireCSA_min'], sig_conn['1']['Wire_WireCSA_max'], 51)

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

def getKey(dct, value):
    """

    :param dct:
    :param value:
    :return:
    """
    return [key for key in dct if (dct[key] == value)]

def category_signal_test(row,col):
    """
    use in demo
    :param row:
    :return:
    """
    if row["cat_"+str(col)][:5]=='BATT_':
        return 1
    elif row["cat_"+str(col)][:4]=='CAN_':
        return 2
    elif row["cat_"+str(col)][:6]=='CANFD_':
        return 3
    elif row["cat_"+str(col)][:4]=='GND_' or 'GND' in row["cat_"+str(col)]:
        return 4
    elif row["cat_"+str(col)][:9]=='GNDR_GND_':
        return 5
    elif row["cat_"+str(col)][:3]=='HV_' or 'POWER' in row["cat_"+str(col)]:
        return 6
    elif row["cat_"+str(col)] == 'na':
        return 8
    else:
        return 7

def return_cat(row, col ,dicc_cat):
    """
    return signal category, use on demo
    :param row:
    :param col:
    :param dicc_cat:
    :return:
    """
    cat=row[col]
    return dicc_cat[cat]

def get_pin_signal_test(sig_conn,i,signal_cat):
    """
    USE IN DEMO

    #1 number pins [0,Num_Pin_unique_max]
    #2 distance to center pin [0,11]bucket
    #3 number of neighbors[0,Num_Pin_unique_max-1]
    #4 internal/external[0,1]
    #5 mean_neighbors [0,11]bucket
    #6 distance centroide [0,11]bucket
    #7 signal category [0,7]
    #8 color wire categorical[0,max_color_categories]
    #9 thickness wire continuous[0,51]bucket
    #10 multicore yes/no [0,1]
    #11 number of internal pins [0,Num_Pin_unique_max//2]
    #12 nosignal [0,1]
    #13 ismulticore [0,1]
    #14 category multicore [0,max_multicore_categories]
    get observation from File
    :param sig_conn:
    :param i:
    :return:
    """
    # create bins
    bins_distance= np.linspace(sig_conn['1']['min_distance'], sig_conn['1']['max_distance'], 11)
    bins_thickness= np.linspace(sig_conn['1']['Wire_WireCSA_min'], sig_conn['1']['Wire_WireCSA_max'], 51)

    Num_Pins=sig_conn[str(i)]['Num_Pins']
    distance_to_center=get_bin_10(random.randint(0, 10), bins_distance)
    neighbors_length=sig_conn[str(i)]['neighbors_length']
    internal_pin = 1 if sig_conn[str(i)]['internal_pin'] =='True' else 0
    mean_neighbors=get_bin_10(random.randint(0, 10), bins_distance)
    centroide_distance=get_bin_10(sig_conn[str(i)]['centroide_distance'], bins_distance)
    signal_group= signal_cat # Signal Category for test
    Cat_Wire_WireColor=sig_conn[str(i)]['Cat_Wire_WireColor']
    WireCSA=get_bin_10(random.randint(0, 51), bins_thickness)
    is_multicore = 0 if sig_conn[str(i)]['is_multicore'] =='na' else 1
    number_internal_pin= sig_conn[str(i)]['number_internal_pin']
    nosignal = random.randint(0, 1)
    ismulticore= random.randint(0, 1)
    cat_multicore=sig_conn[str(i)]['multicore_same']
    return list([Num_Pins,distance_to_center,neighbors_length,internal_pin,mean_neighbors,
                 centroide_distance,signal_group,Cat_Wire_WireColor,WireCSA,is_multicore,
                 number_internal_pin,nosignal,ismulticore, cat_multicore])

def get_models_sizes(connectors_dicc):
    """
    create a vector with all posible sizes of connectors and a dicctionary with key num_pins and a vector
    with the names of connectors that have this pins

    :param connectors_dicc:
    :return:
    """
    d_models = []
    model_connectors = {}

    for connector in connectors_dicc.keys():
        num_pins = connectors_dicc[connector]['1']['Num_Pins']

        if not (num_pins in d_models):
            d_models.append(num_pins)

            model_connectors[num_pins] = []
        if not (connector in model_connectors[num_pins]):
            model_connectors[num_pins].append(connector)
    return d_models, model_connectors


def create_dataframe_test(Num_Pins, num_signals):
    """

    :param Num_Pins:
    :param num_signals:
    :return:
    """
    cols = [str(i) for i in range(1, Num_Pins + 1)]
    data_test = pd.DataFrame(data=[], columns=cols)
    for x in range(0, 50000):
        variant = []
        data = {}
        for i in range(0, Num_Pins):
            variant.append(random.randint(0, num_signals - 1))
            data[str(i + 1)] = random.randint(0, num_signals - 1)
        data_test = data_test.append(data, ignore_index=True)

    for col in data_test.columns:
        data_test["cat_" + str(col)] = data_test.apply(lambda row: return_cat(row, col, dicc_13), axis=1)

        data_test[col] = data_test.apply(lambda row: category_signal_test(row, col), axis=1)

    return data_test

def calculate_max_reward(Num_Pins,  connectors_dicc, connector):
    """

    :param Num_Pins:
    :param num_signals:
    :param dicc_signals:
    :return:
    """


    signal_cats=[]
    for key in connectors_dicc[connector].keys():
        if key != 'PartNumber':
            signal_cats.append(connectors_dicc[connector][key]['signal_group'])

    try:
        num_internal_pins= connectors_dicc[connector]['1']['number_internal_pin']
    except:
        num_internal_pins = 0
    # reward different categories each pin
    reward_signal_cat=Num_Pins * 4
    #reward high voltage close to empty cavities
    if (1 in signal_cats) or (6 in signal_cats) or (8 in signal_cats):
        reward_high_volatge= num_internal_pins * 5
    else:
        reward_high_volatge=0
    # if GROUND is present reward more when far from High Power categories
    if ((1 in signal_cats) or (6 in signal_cats) or (8 in signal_cats)) and ((5 in signal_cats) or (4 in signal_cats )):
        reward_ground = 3
    else:
        reward_ground = 0
    # reward high Power b#not with ground
    if ((4 in signal_cats) or (5 in signal_cats) or (8 in signal_cats)) and ((1 in signal_cats) or (6 in signal_cats )):
        reward_HV = 3
    else:
        reward_HV = 0
    # reward all as must as distance as possible from HP
    if (6 in signal_cats):
        reward_distance= (Num_Pins -1)*3
    else:
        reward_distance = 0

    # rewards from different colors
    reward_color = Num_Pins * 4

    # reward internal pins
    reward_internal_pins= num_internal_pins * 4

    # reward multicore
    multicore = []
    for key in connectors_dicc[connector].keys():
        if key != 'PartNumber':
            multicore.append(connectors_dicc[connector][str(key)]['is_multicore'])

    num_of_multicore=sum(multicore)
    reward_multicore=num_of_multicore * 6

    reward = reward_multicore + reward_internal_pins + reward_color +reward_distance + reward_HV + reward_ground \
        + reward_high_volatge + reward_signal_cat
    return reward

def create_dataframe_test_2(Num_Pins, num_signals, dicc_signals):
    """

    :param Num_Pins:
    :param num_signals:
    :param dicc_signals:
    :return:
    """
    number_possible_variations= 10000
    if num_signals < 10 and Num_Pins < 10:
        number_possible_variations=abs(np.int64(num_signals**Num_Pins))
    else:
        print("huge Number")

    num_iterations=0
    #power= len(str(number_possible_variations)) - 4

    dicc_signals_copy = copy.deepcopy(dicc_signals)

    if "na" in list(dicc_signals.values()):
        key = getKey(dicc_signals_copy, 'na')[0]
        dicc_signals_copy.pop(key, None)

    if number_possible_variations < 5000:
        num_iterations = number_possible_variations
    else:

        num_iterations = 10000

    cols = [str(i) for i in range(1, Num_Pins + 1)]
    data_test = pd.DataFrame(data=[], columns=cols)

    print(f"Dataframe size {num_iterations}")
    for x in range(0, num_iterations):
        signals = list(dicc_signals_copy.values())
        conn = np.zeros((Num_Pins), dtype="int")
        pin_used = []
        breaker = False
        data = {}
        while breaker == False:
            pin = random.randint(0, Num_Pins - 1)
            if not (pin in pin_used):
                pin_used.append(pin)
                if len(signals) > 0:
                    pos = random.randint(0, len(signals) - 1)
                    conn[pin] = getKey(dicc_signals, signals[pos])[0]
                    data[str(pin + 1)] = getKey(dicc_signals, signals[pos])[0]
                    # print(signals[pos])
                    signals.remove(signals[pos])
                else:
                    data[str(pin + 1)] = getKey(dicc_signals, 'na')[0]
                    # print("na")
            if len(pin_used) == Num_Pins:
                data_test = data_test.append(data, ignore_index=True)
                breaker = True


    for col in data_test.columns:
        data_test["cat_" + str(col)] = data_test.apply(lambda row: return_cat(row, col, dicc_signals), axis=1)
        data_test[col] = data_test.apply(lambda row: category_signal_test(row, col), axis=1)

    number_possible_variations = abs(np.int64(num_signals ** Num_Pins))

    return data_test, number_possible_variations, num_iterations

def smap(f):
    return f()

def train_models_multi(d_models, connectors_dicc, data, max_categories, algo, model_connectors):

    pool = multiprocessing.Pool(processes=len(d_models))
    funcs = []
    for size in d_models:
        connector = model_connectors[size][0]
        func = functools.partial(train_agent2, connectors_dicc, connector, data, max_categories, algo, size)
        funcs.append(func)

    print("multiprocessing")
    res = pool.map(smap, funcs)


    pool.close()
    pool.join()
    return res



def predict_demo(chunck, data, algo, connectors_dicc, connector):
    """
    Demo functionality
    :param data_test:
    :param algo:
    :param connectors_dicc:
    :param connector:
    :return:

    """
    # create chunck
    start = chunck[0]
    end = chunck[1]
    data_test = data[start:end]

    data_test['prediction'] = 0
    for i, row in data_test.iterrows():
        model, env = demo_agent2(connectors_dicc, connector, algo)
        obs = env.reset()
        for key in connectors_dicc[connector].keys():
            if key != 'PartNumber':
                # for c in tcols:
                signalcat = data_test.loc[i, key]
                obs = np.array(get_pin_signal_test(connectors_dicc[connector], key, signalcat))
                action, _state = model.predict(obs, deterministic=False)

                obs, reward, done, info = env.step(action.astype(int))

                env.render()
                if done:
                    obs = env.reset()

        #print(reward)
        data_test.at[i, 'prediction'] = reward

    return data_test


def split_dataframe(df, chunk_size = 10000):
    """

    :param df:
    :param chunk_size:
    :return:
    """
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append((i*chunk_size,(i+1)*chunk_size))
    return chunks


def prediction_multiprocessing(data,algo, connectors_dicc, connector):
    """
    multiprocessing Prediction
    :param data:
    :param algo:
    :param connectors_dicc:
    :param connector:
    :return:
    """

    # number of processes
    max_workers = int(multiprocessing.cpu_count()) - 1
    # create chuncks
    chunksize = (len(data) // max_workers) + 1
    chunksdf = split_dataframe(data, chunksize)
    # create Pool
    pool = multiprocessing.Pool(processes=max_workers)
    funcs = []
    for chunck in chunksdf:

        func = functools.partial(predict_demo, chunck, data, algo, connectors_dicc, connector)
        funcs.append(func)

    print("multiprocessing")
    res = pool.map(smap, funcs)

    pool.close()
    pool.join()
    data_final=res[0]
    for i in range(1, len(res)):
        data_final = data_final.append(res[i],  ignore_index=True)

    return data_final





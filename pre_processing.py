import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import numpy as np
import pickle
import os
from math import sqrt
from IPython import embed
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
    elif row["Signal Name"][:4]=='GND_':
        return 4
    elif row["Signal Name"][:9]=='GNDR_GND_':
        return 5
    elif row["Signal Name"][:3]=='HV_':
        return 6
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
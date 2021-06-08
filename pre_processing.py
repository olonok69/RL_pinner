import pandas as pd
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

def get_neighbors_partnumbers(data):
    """

    :param data:
    :return:
    """
    data["distance_to_center"] = data.apply(lambda row: cal_distance_center(row), axis=1)
    list_pn = list(data['Symbol Name'].unique())
    cols = ['Symbol Name', 'Pin Name', 'Pin CenterY', 'Pin CenterX', 'Pin Width',
            'Pin Height', 'distance_to_center', 'neighbors']

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
            dataout = pd.concat([dataout, data1])
        else:
            #print(pn)
            data1['neighbors'] = ["no"]
            dataout = pd.concat([dataout, data1])

    dataout["internal_pin"]="False"
    cols = ['Symbol Name', 'Pin Name', 'Pin CenterY', 'Pin CenterX', 'Pin Width',
            'Pin Height', 'distance_to_center', 'neighbors', 'internal_pin']

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
                print(f"{centroide_distance} {distance_to_center}")
                data1.at[i,'internal_pin']= str((centroide_distance*1.1> distance_to_center>centroide_distance*0.90)
                                                and (distance_to_center>centroide_distance))

            dataout2 = pd.concat([dataout2, data1])
        else:
            #print(pn)
            data1['internal_pin']=str(False)
            dataout2 = pd.concat([dataout2, data1])



    return dataout2


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
    #jpin symbols information, distances and neighbors
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


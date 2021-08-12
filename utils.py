import os
import pickle

dir="c:/temp/"
def save_dict_pickle(path, dict, file):
    """
    save dictionary to pickle file
    :param path:
    :param dict:
    :param file:
    :return:
    """

    with open(os.path.join(dir,path, file), 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def load_dicc(path, file):
    """
    save dictionary to pickle file
    :param path:
    :param dict:
    :param file:
    :return:
    """

    with open(os.path.join(dir, path, file), 'rb') as handle:
        dict= pickle.load(handle)
    return dict
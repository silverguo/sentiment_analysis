from src import *
import configparser
import pickle

# main
if __name__ == '__main__':
    # read the config
    config = configparser.ConfigParser()
    config.read('./config.ini')
    imdbPath = config.get('DATA', 'IMDB_PATH')

    # load data
    dl = ImdbLoader(imdbPath)
    dictReview = dl.dict_load()

    # pickle dict
    with open('./data/temp/imdb_dict.pkl', 'wb') as f:
        pickle.dump(dictReview, f)


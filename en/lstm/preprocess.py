from src import *
import configparser

def main():
    # read the config
    config = configparser.ConfigParser()
    config.read('./config.ini')
    imdbPath = config.get('DATA', 'IMDB_PATH')

    # load data
    dl = ImdbLoader(imdbPath)
    dl.load_set()
    return

if __name__ == '__main__':
    main()
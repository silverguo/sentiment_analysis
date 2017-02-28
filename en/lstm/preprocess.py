from src import *
import configparser

def main():
    # read the config
    config = configparser.ConfigParser()
    config.read('./config.ini')
    trainPath = config.get('DATA', 'IMDB_PATH')

    # load data
    print(trainPath)
    return

if __name__ == '__main__':
    main()
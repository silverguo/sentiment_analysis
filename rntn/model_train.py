from src import *
import configparser

def main():
    # read the configuration file
    config = configparser.ConfigParser()
    config.read('./config.ini')
    trainPath = config.get('DATA', 'STD_TRAIN')

    # load train data
    # lexicon, allTree = dataPrep(trainPath)

    # graph of tensorflow
    model_train()
    return

if __name__ == '__main__':
    # main function
    main()